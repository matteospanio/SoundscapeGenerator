import os
import logging
import argparse
import torch
import wandb
import torch.nn as nn
from accelerate import Accelerator
from env_variables import model_cache_path
from utils.riffusion_pipeline import RiffusionPipeline
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

VERBOSITY_OPTS = {
    "info": logging.INFO,
    "error": logging.ERROR,
    "debug": logging.DEBUG,
}


def add_extra_channel(images):
    extra_channel = torch.zeros(
        images.size(0), 1, images.size(2), images.size(3), device=images.device
    )
    return torch.cat((images, extra_channel), dim=1)


def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to run the finetuning process for.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=["error", "info", "debug"],
        help="The level of logging verbosity",
        default="debug",
    )
    parser.add_argument("-dst", "--destination", help="Where to save the model")


def main():
    parser = argparse.ArgumentParser()
    parse_args(parser)
    args = parser.parse_args()
    epochs = args.epochs
    dst = args.destination

    logging.basicConfig(
        level=VERBOSITY_OPTS[args.verbosity],
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Pad((0, 0, 11, 0)),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    # Initialize the accelerator
    accelerator = Accelerator(mixed_precision="fp16", device_placement=True)

    # Get the process rank (for distributed setups)
    local_rank = int(os.getenv("LOCAL_RANK", 0))  # Rank of the current process

    # Initialize WandB only on the main process (rank 0)
    if local_rank == 0:
        run = wandb.init(
            project="SoundscapeGenerator",
            reinit=True,  # Ensure a new run is started even if a previous one exists
            settings=wandb.Settings(
                start_method="fork"
            ),  # Simplify to avoid issues with multiprocessing
        )
        logger.info("WandB run initialized")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # Print debugging info for all processes
    if local_rank == 0:
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        logger.info(f"Number of devices: {accelerator.num_processes}")
        logger.info(f"Using device: {accelerator.device}")
    logger.debug("Cleaning CUDA cache")
    torch.cuda.empty_cache()

    # Load dataset
    dataset = datasets.ImageFolder(root="categorized_spectrograms", transform=transform)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    if local_rank == 0:
        logger.debug("Dataset ready")

    # Load the RiffusionPipeline
    pipeline = RiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="riffusion/riffusion-model-v1",
        cache_dir=model_cache_path,
        resume_download=True,
    )
    if local_rank == 0:
        logger.debug("Model is loaded")

    # Assuming the pipeline has a model attribute that is trainable
    unet = pipeline.unet

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    if local_rank == 0:
        logger.debug("Started training")

    # Training loop
    num_epochs = epochs  # Set the number of epochs
    for epoch in range(num_epochs):
        if local_rank == 0:
            logging.info(f"Epoch: {epoch}")
        unet.train()  # Set the model to training mode
        running_loss = 0.0

        for batch in dataloader:
            images, labels = batch

            images = add_extra_channel(images)

            timesteps = torch.randint(
                0, 1000, (images.size(0),), device=accelerator.device
            ).long()  # Example timesteps
            encoder_hidden_states = torch.randn(
                images.size(0), 1, 768, device=accelerator.device
            )  # Now (4, 1, 768)
            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = unet(images, timesteps, encoder_hidden_states)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            accelerator.backward(loss)

            # Log only from the main process (rank 0)
            if local_rank == 0:
                wandb.log({"loss": loss.item()})

            optimizer.step()  # Optimize the parameters

            running_loss += loss.item()

        # Print the average loss for this epoch
        if local_rank == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}"
            )

    if local_rank == 0:
        logger.debug("Finished training")

    # Save the trained model (only in the main process)
    if local_rank == 0:
        unet.save_pretrained(dst)

        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(dst)
        run.log_artifact(artifact)

    # Ensure that the WandB run is finished properly (only in main process)
    if local_rank == 0:
        run.finish()


if __name__ == "__main__":
    main()

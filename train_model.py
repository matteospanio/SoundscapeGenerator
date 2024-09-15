import os
import argparse
import logging
import torch
import torch.nn as nn
import wandb
from diffusers import DDPMScheduler
from diffusers import DiffusionPipeline
from torch.optim import AdamW
from torch.utils.data import DataLoader

from env_variables import model_cache_path
from utils.training_utils import CustomImageDataset

logger = logging.getLogger(__name__)

VERBOSITY_OPTS = {
    "info": logging.INFO,
    "error": logging.ERROR,
    "debug": logging.DEBUG,
}


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


def main():

    parser = argparse.ArgumentParser()
    parse_args(parser)
    args = parser.parse_args()
    epochs = args.epochs

    logging.basicConfig(
        level=VERBOSITY_OPTS[args.verbosity],
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not os.path.exists("categorized_spectrograms"):
        logging.error(
            "Categorized spectrograms are missing, run the categorize_spectrograms.py script first"
        )
        return

    # Logging in to wandb
    try:
        run = wandb.init(
            project="SoundscapeGenerator",
        )
    except Exception as e:
        raise Exception(f"Wandb login failed due to {e}")

    # Model loading
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            "riffusion/riffusion-model-v1",
            cache_dir=model_cache_path,
            resume_download=True,
        )

        logging.debug("Model is loaded")
        # Extract model components
        model = pipeline.unet
        model = nn.DataParallel(model)
        text_encoder = pipeline.text_encoder
        tokenizer = pipeline.tokenizer

    except Exception as e:
        raise Exception(f"Model loading failed due to {e}")

    # Dataset loading
    try:
        dataset = CustomImageDataset(root_dir="categorized_spectrograms")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        logger.debug("Dataset loading successful")
    except Exception as e:
        raise Exception(f"Dataset loading failed due to {e}")

    # training
    if not torch.cuda.is_available():
        raise Exception("Cuda is not available, aborted training")

    # Move model to device
    device = "cuda"
    model.to(device)
    text_encoder.to(device)

    logger.info(f"\nThere are {torch.cuda.device_count()} available devices")

    # Training settings
    num_epochs = epochs
    learning_rate = 1e-4
    scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=500)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    logger.debug("Started training")
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch: {epoch}")
        for images, labels in dataloader:
            images = images.to(device)

            # Generate text embeddings
            text_inputs = [f"sndscp {dataset.classes[label]}" for label in labels]
            text_inputs = tokenizer(
                text_inputs, return_tensors="pt", padding=True, truncation=True
            ).input_ids
            text_embeddings = text_encoder(text_inputs.to(device)).last_hidden_state

            # Forward pass
            noise = torch.randn_like(images)
            time_steps = torch.randint(
                0,
                scheduler.num_train_timesteps,
                (images.shape[0],),
                device=images.device,
            ).long()
            noisy_images = scheduler.add_noise(images, noise, time_steps)

            model_output = model(noisy_images, time_steps, text_embeddings)["sample"]

            # Compute loss (simplified)
            loss = torch.nn.functional.mse_loss(model_output, noise)
            wandb.log({"loss": loss.item()})

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

    logger.info("Finished training")

    model.save_pretrained(model_cache_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(model_cache_path)
    run.log_artifact(artifact)

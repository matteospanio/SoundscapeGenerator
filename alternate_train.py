import os
import torch
import wandb
import torch.nn as nn
from accelerate import Accelerator
from env_variables import model_cache_path
from utils.riffusion_pipeline import RiffusionPipeline
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def add_extra_channel(images):
    extra_channel = torch.zeros(images.size(0), 1, images.size(2), images.size(3), device=images.device)
    return torch.cat((images, extra_channel), dim=1)


# Define transformations
transform = transforms.Compose([
    transforms.Pad((0, 0, 11, 0)),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Initialize the accelerator
accelerator = Accelerator(mixed_precision="fp16", device_placement=True)

print(accelerator.is_local_main_process)

# Initialize wandb only in the main process (rank 0)
if accelerator.is_local_main_process:
    try:
        wandb.login(key="0cab68fc9cc47efc6cdc61d3d97537d8690e0379")
        print('Wandb login successful')
        run = wandb.init(
            project="SoundscapeGenerator",
            reinit=True  # Ensure a new run is started even if a previous one exists
        )
    except Exception as e:
        raise Exception(f"Wandb login failed due to {e}")

# Print debugging info for all processes
if accelerator.is_local_main_process:
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Number of devices: {accelerator.num_processes}")
    print(f"Using device: {accelerator.device}")
torch.cuda.empty_cache()

# Load dataset
dataset = datasets.ImageFolder(root='categorized_spectrograms', transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

if accelerator.is_local_main_process:
    print('Dataset ready')

# Load the RiffusionPipeline
pipeline = RiffusionPipeline.from_pretrained(pretrained_model_name_or_path="riffusion/riffusion-model-v1",
                                             cache_dir=model_cache_path,
                                             resume_download=True,
                                             )
if accelerator.is_local_main_process:
    print('Model is loaded')

# Assuming the pipeline has a model attribute that is trainable
unet = pipeline.unet

# Define optimizer and loss function
optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

if accelerator.is_local_main_process:
    print('Started training')

# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    if accelerator.is_local_main_process:
        print(f"Epoch: {epoch}")
    unet.train()  # Set the model to training mode
    running_loss = 0.0

    for batch in dataloader:
        images, labels = batch

        images = add_extra_channel(images)

        timesteps = torch.randint(0, 1000, (images.size(0),), device=accelerator.device).long()  # Example timesteps
        encoder_hidden_states = torch.randn(images.size(0), 1, 768, device=accelerator.device)  # Now (4, 1, 768)
        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = unet(images, timesteps, encoder_hidden_states)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        accelerator.backward(loss)

        # Log only from the main process (rank 0)
        if accelerator.is_local_main_process:
            wandb.log({"loss": loss.item()})

        optimizer.step()  # Optimize the parameters

        running_loss += loss.item()

    # Print the average loss for this epoch
    if accelerator.is_local_main_process:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

if accelerator.is_local_main_process:
    print('Finished training')

# Save the trained model (only in the main process)
if accelerator.is_local_main_process:
    unet.save_pretrained('path/to/save/model')

# Ensure that the WandB run is finished properly (only in main process)
if accelerator.is_local_main_process:
    wandb.finish()

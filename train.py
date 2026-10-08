# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:58:45 2026

@author: jonat
"""

"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import data_setup, engine, model, utils
import argparse
from torchvision import transforms

# Setup hyperparameters
MODEL = "TinyVGG"
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--hidden_units", type=int)
args = parser.parse_args()

if args.model:
    MODEL = args.model
if args.batch_size:
    BATCH_SIZE = args.batch_size
if args.lr:
    LEARNING_RATE = args.lr
if args.num_epochs:
    NUM_EPOCHS = args.num_epochs
if args.hidden_units:
    HIDDEN_UNITS = args.hidden_units
       
# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print(f"Model: {MODEL} Num epochs: {NUM_EPOCHS} Batch size: {BATCH_SIZE} Learning rate: {LEARNING_RATE} Hidden units: {HIDDEN_UNITS}")


# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model_class = getattr(model, MODEL)
model = model_class(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

from timeit import default_timer as timer
start_time = timer()

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

end_time = timer()
print(f"[INFO] Total Training Time: {end_time - start_time: .3f} seconds")

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
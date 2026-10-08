# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:41:37 2026

@author: jonat
"""

# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision
from helper_functions import plot_loss_curves
from torch import nn
from torchvision import transforms, io
from torchinfo import summary
import data_setup, engine
from PIL import Image
from typing import List, Tuple
from pred_and_plot import pred_and_plot_image
from pathlib import Path
import sys

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

#weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
#model = torchvision.models.efficientnet_b0(weights=weights).to(device)

#weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
#model = torchvision.models.mobilenet_v3_small(weights=weights).to(device)

weights = torchvision.models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
model = torchvision.models.resnext101_64x4d(weights=weights)

print(model)

auto_transforms = weights.transforms()
print(auto_transforms)

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms,
                                                                               batch_size=32)

for param in model.parameters():
    param.requires_grad = False
    
output_shape = len(class_names)

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=2048,
                    out_features=output_shape,
                    bias=True)).to(device)

print(summary(model, input_size=(1, 3, 224, 224), col_names=("input_size", "output_size", "num_params", "trainable")))

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=3,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

plot_loss_curves(results)

# Get a random list of image paths from test set
import random
num_images_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

# Make predictions on and plot the images
for image_path in test_image_path_sample:
    pred_and_plot_image(model=model, 
                        image_path=image_path,
                        class_names=class_names,
                        # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                        image_size=(224, 224))

image_file = "sushi-3.jpg"
#import and predict custom image
data_path = Path("data") 
# Setup custom image path
custom_image_path = data_path / image_file
print(custom_image_path)

pred_and_plot_image(model=model, 
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                    image_size=(224, 224))


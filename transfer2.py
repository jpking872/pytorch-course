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
from torchvision import transforms, io, datasets
from torchinfo import summary
import data_setup, engine
from PIL import Image
from typing import List, Tuple
from pred_and_plot import pred_and_plot_image
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
import data_setup

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

auto_transforms = weights.transforms()
print(auto_transforms)

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(train_dir, transform=auto_transforms)
test_data = data_setup.ImageFolderWithPaths(test_dir, transform=auto_transforms)

# Get class names
class_names = train_data.classes

BATCH_SIZE = 1
EPOCHS = 3

# Turn images into data loaders
train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False, # don't need to shuffle test data
)

for param in model.features.parameters():
    param.requires_grad = False
    
output_shape = len(class_names)

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape,
                    bias=True)).to(device)

'''print(summary(model=model,
        input_size=(1, 3, 224, 224), # example of [batch_size, color_channels, height, width]
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]))'''

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()


# Setup training and save the results
# Loop through training and testing steps for a number of epochs
for epoch in range(EPOCHS):
    train_loss, train_acc = engine.train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
    test_loss, test_acc, results = engine.test_step_with_sample(model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device)
    

results['right'] = (results['pred'] == results['label'])
res_sorted = results.sort_values(by=['right', 'prob'], ascending=[True, False])
print(res_sorted)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

#visualize
plt.figure(figsize=(10,1))

i = 1
for idx, row in res_sorted.head(5).iterrows():

    plt.subplot(1, 5, i)
    print(row['path'])
    image_path = Path(row['path'][0])
    img = Image.open(str(image_path))
    plt.imshow(img)
    pred_label = class_names[row['pred']]
    true_label = class_names[row['label']]
    title_text = f"Pred: {pred_label}\nTrue: {true_label}\nProb: {row['prob']:.3f}"
    plt.title(title_text)
    plt.axis(False)
    i+=1


'''plot_loss_curves(results)

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
                    image_size=(224, 224))'''


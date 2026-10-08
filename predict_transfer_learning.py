# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:21:05 2026

@author: jonat
"""

import torch
from torch import nn
import torchvision
import data_setup, engine, model, utils
from pathlib import Path
import argparse
from torchvision import io
from torchvision import transforms

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--image_file", type=str, required=True)
args = parser.parse_args()

image_file = args.image_file

#import and predict custom image
data_path = Path("data") 
# Setup custom image path
custom_image_path = data_path / image_file

custom_image = io.read_image(str(custom_image_path)).type(torch.float32)
custom_image = custom_image / 255

# Print out image data
print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
model_0 = torchvision.models.efficientnet_b0(weights=weights).to(device)

auto_transform = weights.transforms()

model_0.eval()
with torch.inference_mode():
    custom_image_pred = model_0(auto_transform(custom_image).unsqueeze(dim=0)).to(device)
    
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
#custom_image_pred_class = class_names[custom_image_pred_label.cpu()]

# Print out prediction logits
print(f"Prediction logits: {custom_image_pred}")
print(f"Prediction probabilities: {custom_image_pred_probs}")
print(f"Prediction label: {custom_image_pred_label}")
#print(f"Class name: {custom_image_pred_class}")


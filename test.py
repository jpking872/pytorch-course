# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:09:51 2026

@author: jonat
"""

import torch
# Import model_builder.py
import model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
model = model.TinyVGG(input_shape=3,
                              hidden_units=10, 
                              output_shape=3).to(device)
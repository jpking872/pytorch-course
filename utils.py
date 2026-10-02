# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:57:55 2026

@author: jonat
"""

"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import model

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save({"model_state" : model.state_dict(),
              "input_shape" : model.input_shape,
              "output_shape": model.output_shape,
              "hidden_units": model.hidden_units, 
              "model_name": model.class_name},
                f=model_save_path)
  
def load_model(model_file: str):
    model_load_path = Path("models") / model_file
    model_params = torch.load(f=model_load_path)
    model_class = getattr(model, model_params["model_name"])
    loaded_model = model_class(model_params["input_shape"], model_params["hidden_units"], model_params["output_shape"])
    loaded_model.load_state_dict(model_params["model_state"])
    print(f"Loaded model state dict: {loaded_model.state_dict()}")
    
    return loaded_model
    
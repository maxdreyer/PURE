import os

import torch
import yaml
from crp.helper import get_layer_names
from typing import List
from transformers import AutoProcessor


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}
        config["wandb_id"] = os.path.basename(config_path)[:-5]
    return config


def get_layer_names_model(model: torch.nn.Module, model_name: str) -> List[str]:
    """
    Get layer names of a model.
    :param model:   model
    :param model_name:  model name (e.g. vgg16)
    :return:
    """
    if "resnet" in model_name:
        layer_names = get_layer_names(model, [InspectionLayer])
    else:
        raise NotImplementedError
    return layer_names


class InspectionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_pil_list, processor: AutoProcessor = None):
        self.data_pil_list=data_pil_list
        self.processor=processor
        self.length=len(data_pil_list)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        #for key in self.keys:
        sample=self.data_pil_list[idx]
        if self.processor:
            sample=self.processor(images=self.data_pil_list[idx], return_tensors="pt")
            return sample['pixel_values'].squeeze()
        
        return sample.squeeze()
        
            
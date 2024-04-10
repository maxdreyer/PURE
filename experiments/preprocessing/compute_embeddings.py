import argparse
import os

import numpy as np
import torch as torch
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import load_maximization
from crp.visualization import FeatureVisualization
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor, AutoProcessor, FlavaImageModel, CLIPVisionModel

from zennit.core import Composite

from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from utils.helper import load_config, get_layer_names_model, CustomDataset
from utils.lrp_composites import EpsilonPlusFlat
from utils.render import crop_and_mask_images


def get_args():
    parser = argparse.ArgumentParser(description='Compute relevances and activations')
    parser.add_argument('--config_file', type=str,
                        default="configs/imagenet/resnet101_timm.yaml"
                        )
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--layer_name', type=str, default="block_3")
    return parser.parse_args()


def main(model_name,
         ckpt_path,
         dataset_name,
         data_path,
         split,
         layer_name):

    NREF = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(dataset_name)(data_path=data_path, preprocessing=False, split=split)

    print("Dataset loaded")

    model = get_fn_model_loader(model_name)(ckpt_path=ckpt_path, n_class=dataset.num_classes)
    model = model.to(device)
    model.eval()
    print("Model loaded")

    attribution = CondAttribution(model)
    model_CLIP = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor_CLIP = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    processor_DINO = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model_DINO = AutoModel.from_pretrained('facebook/dinov2-base')

    model_CLIP.eval()
    model_DINO.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    layer_names = get_layer_names_model(model, model_name)
    cc = ChannelConcept()
    layer_map = {layer: cc for layer in layer_names}

    fname = f"{model_name}_{dataset_name}_{split}"
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                              path=f"crp_files/{fname}", max_target="max", abs_norm=False)

    d_c_sorted, a, rf_c_sorted = load_maximization(fv.ActMax.PATH, layer_name)
    num_neurons = d_c_sorted.shape[1]

    CLIP = []
    DINO = []
    model_CLIP = model_CLIP.to(device)
    model_DINO = model_DINO.to(device)
    for i, neuron in enumerate(tqdm(np.arange(0, num_neurons))):

        CLIP.append([])
        DINO.append([])

        # print(f"Neuron {neuron} / {i}")
        ref_imgs = fv.get_max_reference([neuron], layer_name, "activation", (0, 50),
                                        composite=composite, rf=True, batch_size=NREF,
                                        plot_fn=crop_and_mask_images)[neuron]
        ref_imgs += fv.get_max_reference([neuron], layer_name, "activation", (50, NREF),
                             composite=composite, rf=True, batch_size=NREF,
                             plot_fn=crop_and_mask_images)[neuron]


        data_set = CustomDataset(ref_imgs, processor_CLIP)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=10, shuffle=False, num_workers=8)
        for inputs in data_loader:
            inputs = inputs.to(device)
            CLIP[i].append(model_CLIP(inputs).last_hidden_state.mean(1).detach().cpu())

        data_set = CustomDataset(ref_imgs, processor_DINO)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=10, shuffle=False, num_workers=8)
        for inputs in data_loader:
            inputs = inputs.to(device)
            DINO[i].append(model_DINO(inputs).last_hidden_state.mean(1).detach().cpu())

        CLIP[i] = torch.cat(CLIP[i], dim=0)
        DINO[i] = torch.cat(DINO[i], dim=0)

    CLIP = torch.stack(CLIP)
    DINO = torch.stack(DINO)

    path = f"results/global_features/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)

    save_file({
        "CLIP": CLIP,
        "DINO": DINO,
    },
        f"{path}/latent_embeddings_{layer_name}_{split}.safetensors")


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    layer_name = args.layer_name
    data_path = config.get('data_path', None)
    ckpt_path = config.get('ckpt_path', None)
    split = args.split

    main(model_name, ckpt_path, dataset_name, data_path, split, layer_name)

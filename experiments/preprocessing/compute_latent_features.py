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

from zennit.core import Composite

from datasets import get_dataset
from models import get_fn_model_loader
from utils.helper import load_config, get_layer_names_model



def get_args():
    parser = argparse.ArgumentParser(description='Compute relevances and activations')
    parser.add_argument('--config_file', type=str,
                        default="configs/imagenet/resnet101_timm.yaml")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--layer_name', type=str, default="block_3")
    return parser.parse_args()


def main(model_name,
         ckpt_path,
         dataset_name,
         data_path,
         split,
         batch_size,
         layer_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(dataset_name)(data_path=data_path, preprocessing=True, split=split)

    print("Dataset loaded")

    model = get_fn_model_loader(model_name)(ckpt_path=ckpt_path, n_class=dataset.num_classes)
    model = model.to(device)
    model.eval()
    print("Model loaded")

    attribution = CondAttribution(model)

    layer_names = get_layer_names_model(model, model_name)
    cc = ChannelConcept()
    layer_map = {layer: cc for layer in layer_names}

    fname = f"{model_name}_{dataset_name}_{split}"
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                              path=f"crp_files/{fname}", max_target="max", abs_norm=False)

    grad_composite = Composite()

    layer_name = layer_name
    prev_layer = layer_names[layer_names.index(layer_name) - 1]

    d_c_sorted, a, rf_c_sorted = load_maximization(fv.ActMax.PATH, layer_name)

    d_c_sorted = d_c_sorted[:, :]
    num_neurons = d_c_sorted.shape[1]

    max_activations = []
    mean_activations = []
    cond_relevances_1 = []
    cond_relevances_2 = []

    for i, neuron in enumerate(tqdm(np.arange(0, num_neurons))):

        max_activations.append([])
        mean_activations.append([])
        cond_relevances_1.append([])
        cond_relevances_2.append([])

        most_act_sample_ids = d_c_sorted[:, neuron]
        dataset_subset = torch.utils.data.Subset(dataset, most_act_sample_ids.flatten())
        dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, num_workers=8)

        for x, _ in dataloader:
            x = x.to(device).requires_grad_()

            attr = attribution(x.requires_grad_(),
                               [{layer_name: neuron}],
                               grad_composite,
                               record_layer=layer_names,
                               start_layer=layer_name,
                               init_rel=lambda act: act.clamp(min=0))

            lower_gradient = attr.relevances[prev_layer].detach().cpu()
            lower_activations = attr.activations[prev_layer].detach().cpu()
            lower_relevance = lower_gradient * lower_activations
            cond_relevances_1[i].append(cc.attribute(lower_relevance, abs_norm=True))

            max_activations[i].append(attr.activations[layer_name].detach().cpu().clamp(min=0).amax((2, 3)))
            mean_activations[i].append(attr.activations[layer_name].detach().cpu().clamp(min=0).mean((2, 3)))

        max_activations[i] = torch.cat(max_activations[i], dim=0)
        mean_activations[i] = torch.cat(mean_activations[i], dim=0)
        cond_relevances_1[i] = torch.cat(cond_relevances_1[i], dim=0)

    max_activations = torch.stack(max_activations)
    mean_activations = torch.stack(mean_activations)
    cond_relevances_1 = torch.stack(cond_relevances_1)

    path = f"results/global_features/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)

    save_file({
        "max_act": max_activations,
        "mean_act": mean_activations,
        "cond_rel": cond_relevances_1,
    },
        f"{path}/latent_features_{layer_name}_{split}.safetensors")


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    batch_size = 40
    data_path = config.get('data_path', None)
    ckpt_path = config.get('ckpt_path', None)
    split = args.split
    layer_name = args.layer_name

    main(model_name, ckpt_path, dataset_name, data_path, split, batch_size, layer_name)

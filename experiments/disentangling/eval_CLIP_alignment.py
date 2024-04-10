import os
from argparse import ArgumentParser
from typing import List

import torch.nn as nn
import torchvision
import umap
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, load_maximization

from crp.visualization import FeatureVisualization
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from safetensors import safe_open
from scipy import stats
from sklearn.cluster import AffinityPropagation, HDBSCAN, KMeans
from torchvision.utils import make_grid
from transformers import CLIPVisionModel, AutoProcessor, FlavaImageModel, FlavaImageConfig, FlavaImageProcessor, \
    FlavaModel, AutoImageProcessor, AutoModel
from zennit.composites import EpsilonPlusFlat
from zennit.core import Composite

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import get_layer_names_model, CustomDataset
from utils.render import vis_opaque_img_border, crop_and_adjust_images, crop_and_mask_images
import torch
import numpy as np
import matplotlib.pyplot as plt
import zennit.image as zimage
from crp.image import imgify

def get_parser(fixed_arguments: List[str] = []):
    parser = ArgumentParser(
        description='Compute and display the top-k most relevant neurons for a given data sample/prediction.', )

    parser.add_argument('--config_file',
                        default="configs/imagenet/resnet101_timm.yaml")
    parser.add_argument('--layer_name', default="block_3", type=str)
    parser.add_argument('--num_clusters', default=2, type=int)
    args = parser.parse_args()

    with open(parser.parse_args().config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["config_name"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args


args = get_parser()
model_name = args.model_name
dataset_name = args.dataset_name

SPLIT = "test"
fv_name = f"crp_files/{model_name}_{dataset_name}_{SPLIT}"
batch_size = 50
n_refimgs = 100
mode = "activation"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split=SPLIT)

ckpt_path = args.ckpt_path

model = get_fn_model_loader(model_name)(n_class=dataset.num_classes, ckpt_path=ckpt_path).to(device)
model.eval()

# model_CLIP = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# processor_CLIP = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
# processor_DINO = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# model_DINO = AutoModel.from_pretrained('facebook/dinov2-base')

# model_DINO = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# processor_DINO = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
# processor_CLIP = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# model_CLIP = AutoModel.from_pretrained('facebook/dinov2-base')

# model_CLIP.eval()
# model_DINO.eval()
#
# model_CLIP = model_CLIP.to(device)
# model_DINO = model_DINO.to(device)


canonizers = get_canonizer(model_name)
composite = EpsilonPlusFlat(canonizers)
cc = ChannelConcept()

layer_names = get_layer_names_model(model, model_name)
layer_map = {layer: cc for layer in layer_names}

print(layer_names)
layer_name = args.layer_name

attribution = CondAttribution(model)

fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                          path=fv_name, max_target="max", abs_norm=False)

if not os.listdir(fv.RelMax.PATH):
    fv.run(composite, 0, len(dataset), batch_size=batch_size)


tensors = {}
path = f"results/global_features/{dataset_name}/{model_name}"
with safe_open(f"{path}/latent_embeddings_{layer_name}_{SPLIT}.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)

CLIP_embeddings = tensors["CLIP"]
DINO_embeddings = tensors["DINO"]

tensors = {}
path = f"results/global_features/{dataset_name}/{model_name}"
with safe_open(f"{path}/latent_features_{layer_name}_{SPLIT}.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)

N = CLIP_embeddings.shape[0]

random_indices = np.random.choice(len(tensors["cond_rel"]), N, replace=False)
neuron_indices = torch.arange(len(tensors["cond_rel"]))[random_indices]
# random_indices = torch.arange(len(tensors["cond_rel"]))
# neuron_indices = torch.arange(len(tensors["cond_rel"]))
n_clusters = args.num_clusters

cond_attributions = tensors["cond_rel"][random_indices][:, :n_refimgs]
clusters = [KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(attr) for attr in cond_attributions]
# counts = [torch.from_numpy(np.unique(c.labels_, return_counts=True)[1]) for c in clusters]
# counts = torch.stack([c if (len(c) == n_clusters) else torch.zeros(n_clusters) for c in counts], dim=0)
# filt_rel = counts.amin(1) > 10

act_attributions = tensors["mean_act"][random_indices][:, :n_refimgs]
act_clusters = [KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(attr) for attr in act_attributions]
# counts = [torch.from_numpy(np.unique(c.labels_, return_counts=True)[1]) for c in act_clusters]
# counts = torch.stack([c if (len(c) == n_clusters) else torch.zeros(n_clusters) for c in counts], dim=0)
# filt = ((counts.amin(1) > 10)*1 + filt_rel*1) >= 1


# neuron_indices = neuron_indices[filt].detach().cpu().numpy().tolist()
# cond_attributions = cond_attributions[filt]
# act_attributions = act_attributions[filt]

inner_rel = []
inter_rel = []
inner_act = []
inter_act = []
inner_CLIP = [] #1787 1805
inter_CLIP = []
inner_dino = []
inter_dino = []
overall_ = []

def compute_distances(CLIP_distances, labels):
    # inter_distances = []
    # inner_distances = []
    labels_ = (labels[None] == labels[:, None])
    labels_ = 1*labels_ - 2*torch.eye(len(labels_)).numpy()
    inter_distances = CLIP_distances[labels_ == 0].flatten().tolist()
    inner_distances = CLIP_distances[labels_ == 1].flatten().tolist()
    if len(inter_distances) == 0:
        inter_distances = inner_distances
    if len(inner_distances) == 0:
        inner_distances = inter_distances
    # for label in np.unique(labels):
    #     cluster_indices = np.where(labels == label)[0]
    #     not_cluster_indices = np.where(labels > label)[0]
    #     inner_distances.extend(CLIP_distances[cluster_indices][:, cluster_indices].flatten().tolist())
    #     inter_distances.extend(CLIP_distances[cluster_indices][:, not_cluster_indices].flatten().tolist())
    return np.mean(inner_distances), np.mean(inter_distances)

for i, neuron in enumerate(neuron_indices):
    print(f"Neuron {neuron} / {i}")
    # ref_imgs = fv.get_max_reference([neuron], layer_name, mode, (0, 50),
    #                                 composite=composite, rf=True, batch_size=n_refimgs, plot_fn=crop_and_mask_images)[neuron]
    #
    # ref_imgs += fv.get_max_reference([neuron], layer_name, mode, (50, n_refimgs),
    #                                 composite=composite, rf=True, batch_size=n_refimgs, plot_fn=crop_and_mask_images)[neuron]
    # data_set = CostumDataset(ref_imgs, processor_CLIP)
    # data_loader = torch.utils.data.DataLoader(data_set, batch_size=n_refimgs, shuffle=False, num_workers=8)

    # CLIP_embeddings = []
    # DINO_embeddings = []
    # for inputs in data_loader:
    #     inputs = inputs.to(device)
    #     CLIP_embeddings.append(model_CLIP(inputs).last_hidden_state.mean(1).detach().cpu())
    #     # CLIP_embeddings.append(model_CLIP(inputs).pooler_output.detach().cpu())
    #
    # data_set = CostumDataset(ref_imgs, processor_DINO)
    # data_loader = torch.utils.data.DataLoader(data_set, batch_size=n_refimgs, shuffle=False, num_workers=8)
    # for inputs in data_loader:
    #     inputs = inputs.to(device)
    #     DINO_embeddings.append(model_DINO(inputs).last_hidden_state.mean(1).detach().cpu())

    # CLIP_embeddings = torch.cat(CLIP_embeddings, dim=0)
    CLIP_cluster = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(CLIP_embeddings[neuron])
    distances = torch.norm(CLIP_embeddings[neuron][None] - CLIP_embeddings[neuron][:, None], dim=2)
    # CLIP_cluster = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(CLIP_embeddings)
    # distances = torch.norm(CLIP_embeddings[None] - CLIP_embeddings[:, None], dim=2)

#     DINO_embeddings = torch.cat(DINO_embeddings, dim=0)
    DINO_cluster = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(DINO_embeddings[neuron])
    # DINO_cluster = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(DINO_embeddings)

    inner_distances, inter_distances = compute_distances(distances, clusters[i].labels_)
    print(f"REL Mean distance inside clusters: {np.mean(inner_distances)}")
    print(f"REL Mean distance inbetween clusters: {np.mean(inter_distances)}")
    inner_rel.append(inner_distances)
    inter_rel.append(inter_distances)
    overall_.append(distances.sum().item() / (distances.shape[-1]**2 - distances.shape[-1] + 1e-8))

    inner_distances, inter_distances = compute_distances(distances, act_clusters[i].labels_)
    print(f"ACT Mean distance inside clusters: {np.mean(inner_distances)}")
    print(f"ACT Mean distance inbetween clusters: {np.mean(inter_distances)}")
    inner_act.append(inner_distances)
    inter_act.append(inter_distances)

    inner_distances, inter_distances = compute_distances(distances, CLIP_cluster.labels_)
    print(f"CLIP Mean distance inside clusters: {inner_distances}")
    print(f"CLIP Mean distance inbetween clusters: {inter_distances}")
    inner_CLIP.append(inner_distances)
    inter_CLIP.append(inter_distances)

    # inner_distances, inter_distances = compute_distances(distances, np.random.randint(0, n_clusters, len(CLIP_cluster.labels_)))
    inner_distances, inter_distances = compute_distances(distances, DINO_cluster.labels_)
    print(f"DINO Mean distance inside clusters: {inner_distances}")
    print(f"DINO Mean distance inbetween clusters: {inter_distances}")
    inner_dino.append(inner_distances)
    inter_dino.append(inter_distances)

print(f"CLIP Intra mean distance: {np.mean(inner_CLIP)}")
print(f"CLIP Inter mean distance: {np.mean(inter_CLIP)}")
print(f"REL Intra mean distance: {np.mean(inner_rel)}")
print(f"REL Inter mean distance: {np.mean(inter_rel)}")
print(f"ACT Intra mean distance: {np.mean(inner_act)}")
print(f"ACT Inter mean distance: {np.mean(inter_act)}")
print(f"DINO Intra mean distance: {np.mean(inner_dino)}") 
print(f"DINO Inter mean distance: {np.mean(inter_dino)}")

vals_full = {
    "CLIP": [inner_CLIP, inter_CLIP],
    "PURE": [inner_rel, inter_rel],
    "Activation": [inner_act, inter_act],
    "DINO": [inner_dino, inter_dino],
    "overall": overall_,
    "neuron_indices": neuron_indices,
}

os.makedirs(f"results/interpretability/{dataset_name}/{model_name}", exist_ok=True)
torch.save(vals_full, f"results/interpretability/{dataset_name}/{model_name}/interpretability_{n_clusters}clusters_{layer_name}_{SPLIT}.pt")

vals = {
    "CLIP": [np.mean(inner_CLIP), np.mean(inter_CLIP), np.std(inner_CLIP) / len(inner_CLIP),
             np.std(inter_CLIP) / len(inter_CLIP)],
    "PURE": [np.mean(inner_rel), np.mean(inter_rel), np.std(inner_rel) / len(inner_rel),
             np.std(inter_rel) / len(inter_rel)],
    "Activation": [np.mean(inner_act), np.mean(inter_act), np.std(inner_act) / len(inner_act),
                   np.std(inter_act) / len(inter_act)],
    "DINO": [np.mean(inner_dino), np.mean(inter_dino), np.std(inner_dino) / len(inner_dino),
                   np.std(inter_dino) / len(inter_dino)],
}

plt.rcParams['text.usetex'] = False
plt.figure(dpi=300)

x = ["CLIP", "DINO", "PURE", "Activation"]

overall = np.mean(overall_)

plt.bar(x, height=[vals[m][1] for m in x], label="inter", color="#E33FDD")
plt.bar(x, height=[vals[m][0] for m in x], label="inner", color="#0094FF")
for i, m in enumerate(x):
    plt.text(i, vals[m][0], f"${vals[m][0]:.2f}\pm{vals[m][2]:.2f}$", ha="center", va="bottom")
    plt.text(i, vals[m][1], f"${vals[m][1]:.2f}\pm{vals[m][3]:.2f}$", ha="center", va="bottom")
    # plt.text(i, vals[m][0], f"{vals[m][0]:.2f}", ha="center", va="bottom")

plt.legend(loc="lower right")
plt.ylabel("distance of CLIP embeddings")
plt.show()





import os
from argparse import ArgumentParser
from typing import List

import torchvision
import umap
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import load_maximization

from crp.visualization import FeatureVisualization
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from safetensors import safe_open
from scipy import stats
from sklearn.cluster import KMeans
from torchvision.utils import make_grid
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import get_layer_names_model
from utils.render import vis_opaque_img_border
import torch
import numpy as np
import matplotlib.pyplot as plt
import zennit.image as zimage


def get_parser(fixed_arguments: List[str] = []):
    parser = ArgumentParser(
        description='Compute and display the top-k most relevant neurons for a given data sample/prediction.', )

    parser.add_argument('--config_file',
                        default="configs/imagenet/resnet50_timm.yaml")
    parser.add_argument('--neurons',
                        default="1,2,3,4,5")
    parser.add_argument('--embeddings',
                        default="pure") # "pure", "CLIP", "activations"
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
batch_size = 100
n_refimgs = 20
mode = "activation"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = get_dataset(dataset_name)(data_path=args.data_path, preprocessing=False, split=SPLIT)

ckpt_path = args.ckpt_path

model = get_fn_model_loader(model_name)(n_class=dataset.num_classes, ckpt_path=ckpt_path).to(device)
model.eval()
canonizers = get_canonizer(model_name)
composite = EpsilonPlusFlat(canonizers)
cc = ChannelConcept()

layer_names = get_layer_names_model(model, model_name)
layer_map = {layer: cc for layer in layer_names}

print(layer_names)
layer_name = layer_names[-1]

attribution = CondAttribution(model)

fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                          path=fv_name, max_target="max", abs_norm=False)

if not os.listdir(fv.RelMax.PATH):
    fv.run(composite, 0, len(dataset), batch_size=batch_size)


d_c_sorted, a, rf_c_sorted = load_maximization(fv.ActMax.PATH, layer_name)


tensors = {}
path = f"results/global_features/{dataset_name}/{model_name}"

if args.embeddings == "pure":
    with safe_open(f"{path}/latent_features_{layer_name}_{SPLIT}.safetensors", framework="pt", device="cpu") as f:
       for key in f.keys():
           tensors[key] = f.get_tensor(key)
    embeddings = tensors["cond_rel"][:, :n_refimgs]
elif args.embeddings == "CLIP":
    with safe_open(f"{path}/latent_embeddings_{layer_name}_{SPLIT}.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    embeddings = tensors["CLIP"][:, :n_refimgs]
elif args.embeddings == "activations":
    with safe_open(f"{path}/latent_features_{layer_name}_{SPLIT}.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    embeddings = tensors["mean_act"][:, :n_refimgs]


n_clusters = 2

neurons = torch.tensor([int(i) for i in args.neurons.split(",")])

for inds_ in [neurons]:
    fig, axs = plt.subplots(2 + n_clusters, len(neurons), dpi=300, figsize=(len(neurons) * 4/1.3, 6/1.4),
                            gridspec_kw={'height_ratios': [len(neurons), 1, *np.ones(n_clusters).tolist()]},)
    for i, inds in enumerate(inds_):
        print(i)
        ref_imgs = fv.get_max_reference([inds.item()], layer_name, mode, (0, n_refimgs),
                                        composite=composite, rf=True, batch_size=n_refimgs,
                                        plot_fn=vis_opaque_img_border)

        embedding = umap.UMAP(n_neighbors=6, min_dist=0.3, spread=1.0)
        X = embedding.fit_transform(embeddings[inds])
        x, y = X[:, 0], X[:, 1]
        xmin = x.min() - 0.2 * (x.max() - x.min())
        xmax = x.max() + 0.2 * (x.max() - x.min())
        ymin = y.min() - 0.2 * (y.max() - y.min())
        ymax = y.max() + 0.2 * (y.max() - y.min())
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values, 0.5)
        Z = np.reshape(kernel(positions).T, X.shape).T
        axs[0][i].contour(Z, extent=[xmin, xmax, ymin, ymax], cmap="Greys", alpha=0.3, extend='min', vmax=Z.max() * 1, zorder=0)
        axs[0][i].scatter(x, y, alpha=0.7, c="black", s=10)

        for j, img_ in enumerate(ref_imgs[inds.item()]):
            imagebox = OffsetImage(img_.resize((100, 100)), zoom=0.15)
            ab = AnnotationBbox(imagebox, (x[j], y[j]), frameon=True, pad=0)
            axs[0][i].add_artist(ab)

        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        axs[0][i].text(0.02, 0.98, f'#{inds.item()}', ha='left', va='top', transform=axs[0][i].transAxes)

        resize = torchvision.transforms.Resize((150, 150))

        NUM = 8
        ref_imgs_ = ref_imgs[inds.item()]
        grid = make_grid(
            [resize(torch.from_numpy(np.asarray(k)).permute((2, 0, 1))) for k in ref_imgs_[:NUM]],
            nrow=NUM,
            padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        axs[1][i].imshow(grid)
        axs[1][i].set_xticks([])
        axs[1][i].set_yticks([])
        axs[1][i].set_ylabel("all") if i == 0 else None



        cluster = KMeans(n_clusters=n_clusters, n_init=20, random_state=123).fit(embeddings[inds])
        labels = np.array(cluster.labels_)
        for lab in np.unique(labels):
            ref_imgs_cluster = [r for k, r in enumerate(ref_imgs_) if labels[k] == lab]
            grid = make_grid(
                [resize(torch.from_numpy(np.asarray(k)).permute((2, 0, 1))) for k in ref_imgs_cluster[:NUM]],
                nrow=NUM,
                padding=0)
            grid = np.array(zimage.imgify(grid.detach().cpu()))
            axs[2 + lab][i].imshow(grid)
            axs[2 + lab][i].set_xticks([])
            axs[2 + lab][i].set_yticks([])
            axs[2 + lab][i].set_ylabel(f"{lab + 1}") if i == 0 else None

    plt.tight_layout()

    path = f"results/neuron_plots/{dataset_name}/{model_name}"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/neurons_{'_'.join([str(n.item()) for n in neurons])}.pdf", dpi=300)

    plt.show()


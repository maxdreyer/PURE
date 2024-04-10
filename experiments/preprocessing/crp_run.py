import os
from argparse import ArgumentParser

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.image import *
from crp.visualization import FeatureVisualization
from torch import nn
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_fn_model_loader, get_canonizer
from utils.helper import load_config, get_layer_names_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config_file", default="configs/imagenet/resnet101_timm.yaml")
    return parser.parse_args()


def main(model_name, ckpt_path, dataset_name, data_path, batch_size, fname):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    SPLIT = "test"
    dataset = get_dataset(dataset_name)(data_path=data_path,
                                        preprocessing=False,
                                        split=SPLIT, )

    model = get_fn_model_loader(model_name)(n_class=dataset.num_classes, ckpt_path=ckpt_path).to(device)
    model.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_names = get_layer_names_model(model, model_name)
    print(layer_names)
    layer_map = {layer: cc for layer in layer_names}

    attribution = CondAttribution(model)

    os.makedirs(f"crp_files", exist_ok=True)

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.preprocessing,
                              path=f"crp_files/{fname}_{SPLIT}", max_target="max", abs_norm=False)

    fv.ActMax.SAMPLE_SIZE = 100
    fv.run(composite, 0, len(dataset), batch_size=batch_size)


if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file)

    model_name = config['model_name']
    dataset_name = config['dataset_name']
    batch_size = config.get('batch_size', 32)

    data_path = config.get('data_path', None)
    ckpt_path = config.get('ckpt_path', None)

    fname = f"{model_name}_{dataset_name}"

    main(model_name, ckpt_path, dataset_name, data_path, batch_size, fname)

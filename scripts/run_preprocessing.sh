python3 -m experiments.preprocessing.crp_run --config_file "configs/imagenet/resnet50_timm.yaml"
python3 -m experiments.preprocessing.crp_run --config_file "configs/imagenet/resnet34_timm.yaml"
python3 -m experiments.preprocessing.crp_run --config_file "configs/imagenet/resnet101_timm.yaml"

python3 -m experiments.preprocessing.compute_latent_features --config_file "configs/imagenet/resnet50_timm.yaml"
python3 -m experiments.preprocessing.compute_latent_features --config_file "configs/imagenet/resnet34_timm.yaml"
python3 -m experiments.preprocessing.compute_latent_features --config_file "configs/imagenet/resnet101_timm.yaml"

python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenet/resnet50_timm.yaml"
python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenet/resnet34_timm.yaml"
python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenet/resnet101_timm.yaml"
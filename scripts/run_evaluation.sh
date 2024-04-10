
experiment=experiments.disentangling.eval_CLIP_alignment

for n_clusters in {2..5}; do
    python3 -m $experiment --config_file "configs/imagenet/resnet50_timm.yaml" --num_clusters $n_clusters
    python3 -m $experiment --config_file "configs/imagenet/resnet34_timm.yaml" --num_clusters $n_clusters
    python3 -m $experiment --config_file "configs/imagenet/resnet101_timm.yaml" --num_clusters $n_clusters
done

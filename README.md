CUDA_VISIBLE_DEVICES=3 python run_evaluation.py --config configs/cra/ft_cra_scannet_scene0376.yaml --model-path out/ft_cra_scannet_scene0376/model_best.pth

CUDA_VISIBLE_DEVICES=1 python run_training.py --config configs/cra/train_with_depth.yaml
CUDA_VISIBLE_DEVICES=1 python run_training.py --config configs/cra/train_with_pretrain.yaml


CUDA_VISIBLE_DEVICES=2 python run_training.py --config configs/cra/ft_cra_scannet_scene0376.yaml
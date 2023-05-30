CUDA_VISIBLE_DEVICES=3 python run_evaluation.py --config configs/cra/ft_cra_scannet_scene0376.yaml --model-path out/ft_cra_scannet_scene0376/model_best.pth

CUDA_VISIBLE_DEVICES=1 python run_training.py --config configs/cra/train_with_depth.yaml

CUDA_VISIBLE_DEVICES=0 python run_training.py --config configs/cra/no_semantic.yaml


CUDA_VISIBLE_DEVICES=2 python run_training.py --config configs/cra/ft_cra_scannet_scene0376.yaml

CUDA_VISIBLE_DEVICES=1 python run_training.py --config configs/cra/train_with_semantic_fpn.yaml
CUDA_VISIBLE_DEVICES=2 python run_training.py --config configs/cra/train_with_maskformer.yaml
CUDA_VISIBLE_DEVICES=3 python run_training.py --config configs/cra/train_with_semantic_fpn_0_1.yaml
CUDA_VISIBLE_DEVICES=4 python run_training.py --config configs/cra/train_cra_scannet.yaml
CUDA_VISIBLE_DEVICES=5 python run_training.py --config configs/cra/only_train_semantic_fpn.yaml
CUDA_VISIBLE_DEVICES=6 python run_training.py --config configs/cra/only_train_resunet.yaml
CUDA_VISIBLE_DEVICES=6 python run_training.py --config configs/cra/train_with_resunet.yaml

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
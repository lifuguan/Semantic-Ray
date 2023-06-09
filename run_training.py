import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/train/gen/neuray_gen_depth_train.yaml')
flags = parser.parse_args()

trainer = Trainer(load_cfg(flags.config))
trainer.run()

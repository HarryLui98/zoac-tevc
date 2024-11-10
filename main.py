import sys
import time
import os
from algo.learner import ZOACLearner
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import ray

@hydra.main(version_base=None, config_path="conf", config_name="config_zoac")
def run_algo(cfg):
    learner = ZOACLearner(cfg)
    learner.train()

if __name__ == '__main__':
    ray.init(local_mode=False, include_dashboard=False)
    run_algo()
    ray.shutdown()

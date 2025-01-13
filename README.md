## Zeroth-Order Actor-Critic: An Evolutionary Framework for Sequential Decision Making

### Prerequisites
The results in the paper is based on the following suggested environment: 
* Python == 3.8.18
* [SUMO](https://sumo.dlr.de/docs/Downloads.php) == 2.3.7
* [traci](https://sumo.dlr.de/docs/TraCI.html#using_traci) == 1.21.0
* [gops](https://gops.readthedocs.io/en/latest/introduction.html) == 1.1.0
* [nevergrad](https://facebookresearch.github.io/nevergrad/) == 0.10.0
* [gymnasium](https://github.com/Farama-Foundation/Gymnasium) == 0.28.1
* [torch](https://pytorch.org) == 2.0.1
* [casadi](https://web.casadi.org) == 3.6.3
* [ray](https://www.ray.io) == 2.9.1
* [hydra-core](https://hydra.cc/docs/intro/) == 1.3.2

### How to Use

1. We use ```hydra``` to configure hyperparameters. See ```conf/config_zoac.yaml``` for details;
2. Simply run ```python main.py```.

### Reference

```
@article{lei2025zeroth,
  title={Zeroth-Order Actor-Critic: An Evolutionary Framework for Sequential Decision Making},
  author={Lei, Yuheng and Lyu, Yao and Zhan, Guojian and Zhang, Tao and Li, Jiangtao and Chen, Jianyu and Li, Shengbo Eben and Zheng, Sifa},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2025},
  publisher={IEEE}
}
```
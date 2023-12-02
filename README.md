# 285-project

**Authors:** Alina Trinh, Samarth Goel, Jonathan Guo

## TODO:

- [jonathan] rework expert data gathering using datasets from sota model githubs
- [alina] implement/copy one of [dqfd, dapg] algorithms from github
- [samarth] rework code to use packages
  - stable baselines 3
  - DL-Engine

Next Steps:

1. test to see if we can get baselines working (no pruning)
2. [done] plug in pruning and see the changes
3. ablation studies on pruning types
   - entropy, variance, etc...
   - l2 vs cos distance, etc...
   - expert optimality (beta?)
4. [extra] ablation studies on pruning hyperparameters

## Relevant Resources

- **General Purpose Deep RL Libraries to Use:**

  - [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
    Contains standard algorithms and ways to evaluate models

  - [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
    Contains pretrained agents and optimized hyperparams for models

- **Specialized Libraries to Use:**

  - [DL Engine](https://github.com/opendilab/DI-engine)
    Has many, many algorithms and a ton of environments to choose from

  - [RL for MuJoCo](https://github.com/aravindr93/mjrl)
    Has DAPG implementation

- **Papers and Algos to Look at:**

  - [DQFD Paper](https://arxiv.org/pdf/1704.03732.pdf)
    Implementations:

    - TF Implementation: [https://github.com/go2sea/DQfD](https://github.com/go2sea/DQfD)
    - TF Implementation: [https://github.com/felix-kerkhoff/DQfD](https://github.com/felix-kerkhoff/DQfD)

  - [DAPG Paper](https://www.roboticsproceedings.org/rss14/p49.pdf)
    Implementations:
    - Data and Dataloading Implementation: [https://github.com/aravindr93/hand_dapg](https://github.com/aravindr93/hand_dapg)

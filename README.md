# 285-project

**Authors:** Alina Trinh, Samarth Goel, Jonathan Guo

## TODO:

- [alina] implement/copy dqfd algorithm from github

- [alina] Fix filtering logic (currently eliminates groups, should prune groups)

<!-- - [samarth] set up logging to compare pruning vs no pruning -->

<!-- - [samarth] set up consistent randomness -->

- [Jonathan] alter expert data generation to create good but not great (suboptimal) demos

  - re-initialize expert weights somehow for each trajectory

- [Jonathan] start on the math for the paper

  - From Sergei: must formalize π\*(a|s) somehow; aka formalize the assumption we're making about the expert

Next Steps:

1. ablation studies on pruning types
   - entropy, variance, etc...
   - l2 vs cos distance, etc...
   - expert optimality (beta?)
2. [extra] ablation studies on pruning hyperparameters

## Relevant Resources

- **General Purpose Deep RL Libraries to Use:**

  - [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/): Contains standard algorithms and ways to evaluate models

  - [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo): Contains pretrained agents and optimized hyperparams for models

- **Papers and Algos to Look at:**

  - [DQFD Paper](https://arxiv.org/pdf/1704.03732.pdf)

    Implementations:

    - TF Implementation: [https://github.com/go2sea/DQfD](https://github.com/go2sea/DQfD)
    - TF Implementation: [https://github.com/felix-kerkhoff/DQfD](https://github.com/felix-kerkhoff/DQfD)

  - [DAPG Paper](https://www.roboticsproceedings.org/rss14/p49.pdf)

    Implementations:

    - Data and Dataloading Implementation: [https://github.com/aravindr93/hand_dapg](https://github.com/aravindr93/hand_dapg)

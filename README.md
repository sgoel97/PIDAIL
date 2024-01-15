# PIDAIL: Pruning Inconsistent Demonstrations for Augmented Imitation Learning

**Authors:** Alina Trinh, Samarth Goel, Jonathan Guo

## Overview

Imitation learning serves as a baseline for expert-based reinforcement learning techniques and is at the forefront of several cutting-edge research problems. Real-world uses of imitation learning, however, can suffer from a lack of high-quality expert data and even from suboptimal expert data, where experts may not make correct decisions or be in disagreement with one another. How to best handle these suboptimal demonstrations is an active, open area of research; in this report, we present the first (to the best of our knowledge) approach that prunes suboptimal demonstrations from the training set before starting the imitation learning process.


We introduce Pruning Inconsistent Demonstrations for Augmented Imitation Learn- ing (PIDAIL), a technique meant to distill an expert dataset into its most useful components. PIDAIL operates under the assumption that while some demonstrators may understand at a high level what an ideal policy is for an environment, they can still be inconsistent when actually giving demonstrations, either because they are indifferent about which actions to take in certain states or because they are uncertain or indecisive about which specific action is the best to take, both leading to potentially suboptimal demonstrations. With PIDAIL, we use heuristics such as dispersion (variance or entropy) or frequency to identify which actions are most likely to be suboptimal and prune the corresponding demonstrations from the agent’s training set. We accomplish this through a variety of observation-clustering algorithms (agglomerative and k-means) and pruning algorithms based on cluster dispersion, outcome dispersion, action frequency, and action value.


We tested our pruning methods on several different environments from OpenAI Gym, ensuring both discrete and continuous action spaces, using four different downstream imitation learning-based algorithms: behavior cloning (BC), deep Q- learning from demonstrations (DQfD), generative adversarial imitation learning (GAIL), and soft Q imitation learning (SQIL). Our results indicate that PIDAIL can enable faster learning and higher returns for almost all environments. Overall, since PIDAIL allows the agent to separate good demonstrations from bad demonstrations before training without needing additional environmental interactions, it can save time, effort, cost, and computational resources.

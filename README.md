# Reinforcement-learning-python

In this project, three different reinforcement learning algorithms (value iteration, policy iteration and Q-learning) were used to solve two different Markov Decision Process (MDP) problems. The results and performance of these algorithms with the different MDP problems were analyzed and compared.

### Description of the problems
1. Frozen Lake
The Forest Lake  is a typical grid-world environment. The left-up grid is the starting state(S) and the right-down grid is the goal state(G). Other grids are either a frozen state (F) which is safe, or a hole state(H) that the agent needs to avoid. The goal of the agent is to reach the goal state G from the start state S without visiting the hole state H. The episode ends when the agent reaches the goal or falls in a hole. Reward is 1 if the agent reaches G and 0 otherwise.
2. The Forest management is a non-grid world environment. The objective of the forest management is to find an optimal policy that maximizes the benefits while taking into account probability of fire occurrence. There are two possible actions: Wait and Cut. Each year there is a probability of the forest been burnt by the fire. Here I used the example module of the MDPtoolbox[2] to model this problem. A transition probability (A x S x S) array P and a reward (S x A) matrix R models this problem. {0, 1…S-1} are the states of the forest, with S-1 being the oldest. “Wait” is action 0 and “Cut” is action 1. After a fire, the forest goes to state 0. 

### Algorithms and Methodology
The three algorithms are implemented by python. For value iteration and policy iteration, the discount factor (Gamma) is varied to observe the impact on the convergence, computation time, and performance. For Q learning, the learning rate is varied to compare the performances.

### Instructions for running the code:
1. frozen lake problems
  run the python file FrozenLake.py  to plot graphs and get the statistical data for frozen lake problems. Graphs will be shown in the figs folder
  
2. forest management problems
  run the python file ForestManagement.py  to plot graphs and get the statistical data for forest management problems. Graphs will be shown in the figs folder.

### Code dependencies:
Python3.7
mdptoolbox-hiive ==4.0.3.1
scikit-learn == 0.19.2
matplotlib == 0.1.3
numpy == 1.15.1
pandas == 0.23.4
gym == 0.10.5


from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
from hiive.mdptoolbox.example import forest
import numpy as np
from numpy.random import choice
from time import time
np.random.seed(1)
P_1, R_1 = forest(S=25, r1=4, r2=2, p=0.1)
P_2, R_2 = forest(S=2000, r1=40, r2=20, p=0.1)


def test_policy(P, R, policy, test_count=1000, gamma=0.95):
    num_state = P.shape[-1]
    total_episode = num_state * test_count
    # start in each state
    total_reward = 0
    for state in range(num_state):
        state_reward = 0
        for state_episode in range(test_count):
            episode_reward = 0
            disc_rate = 1
            while True:
                # take step
                action = policy[state]
                # get next step using P
                probs = P[action][state]
                candidates = list(range(len(P[action][state])))
                next_state = choice(candidates, 1, p=probs)[0]
                # get the reward
                reward = R[state][action] * disc_rate
                episode_reward += reward
                # when go back to 0 ended
                disc_rate *= gamma
                if next_state == 0:
                    break
            state_reward += episode_reward
        total_reward += state_reward
    return total_reward / total_episode


# value iteration
vi_1 = ValueIteration(P_1, R_1, gamma=0.95, max_iter=100000)
vi_1.run()
vi_1_reward = test_policy(P_1, R_1, vi_1.policy)
print(vi_1.iter)
print(vi_1.time)
print(vi_1_reward)
print(vi_1.policy)

vi_2 = ValueIteration(P_2, R_2, gamma=0.95, max_iter=100000)
vi_2.run()
vi_2_reward = test_policy(P_2, R_2, vi_2.policy)
print(vi_2.iter)
print(vi_2.time)
print(vi_2_reward)
print(vi_2.policy)

# Policy iteration
pi_1 = PolicyIteration(P_1, R_1, gamma=0.95, max_iter=100000)
pi_1.run()
pi_1_reward = test_policy(P_1, R_1, pi_1.policy)
print(pi_1.iter)
print(pi_1.time)
print(pi_1_reward)
print(pi_1.policy)

pi_2 = PolicyIteration(P_2, R_2, gamma=0.95, max_iter=100000)
pi_2.run()
pi_2_reward = test_policy(P_2, R_2, pi_2.policy)
print(pi_2.iter)
print(pi_2.time)
print(pi_2_reward)
print(pi_2.policy)

# Q learning
t1 = time()
q_1 = QLearning(P_1, R_1, gamma=0.999, alpha=0.1, alpha_decay=0.99999, epsilon=1, epsilon_decay=0.9999, n_iter=1000000)
q_1.run()
q_1_reward = test_policy(P_1, R_1, q_1.policy)
print(time() - t1)
print(q_1_reward)
print(q_1.policy)

t2 = time()
q_2 = QLearning(P_2, R_2, gamma=0.999, alpha=0.1, alpha_decay=0.99999, epsilon=1, epsilon_decay=0.9999, n_iter=1000000)
q_2.run()
q_2_reward = test_policy(P_2, R_2, q_2.policy)
print(time() - t2)
print(q_2_reward)
print(q_2.policy)

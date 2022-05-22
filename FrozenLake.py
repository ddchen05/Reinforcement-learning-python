from time import time
import gym
import matplotlib.pyplot as ply
import numpy as np
import pandas as pd
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "20x20": [
        "SFFFFFFHHHFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFHHFF",
        "FFFHFFFFFFFHHFFFFFFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFHFFFFFFFHHFF",
        "FFFFFHFFFFHHFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFFFFHHHHHHHFF",
        "HHHHFHFFFFFFFFFFHHFF",
        "FFFFFHFFFFHHHFFFHHFF",
        "FFFFFFFFFFFFFFFFHHFF",
        "FFFFFHFFFFFFHFFFHHFF",
        "FFFFFHFFFFFFFFFFHHFF",
        "FFFFFFFFFFFHFFFFFFFF",
        "FHHFFFHFFFFHFFFFFHFF",
        "FHHFHFHFFFFFFFFFFFFF",
        "FFFHFFFFFHFFFFHHFHFG"
    ],
}

frozenLake4x4 = gym.make('FrozenLake-v0').env
frozenLake20x20 = FrozenLakeEnv(desc=MAPS["20x20"])
frozenLake4x4.render()
frozenLake20x20.render()

TERM_STATE_MAP = {"4x4": [5, 7, 11, 12],
                  "20x20": [7, 8, 9, 36, 37, 43, 51, 52, 65, 76, 77, 85, 96, 97, 105, 116, 117, 128, 136, 137, 145, 150,
                            151, 156, 157, 165, 176, 177, 185, 196, 197, 211, 212, 213, 214, 215, 216, 217, 220, 221,
                            222,223, 225, 236, 237, 245, 250, 251, 252, 256, 257, 276, 277, 285, 292, 296, 297, 305, 326,
                            317, 331, 341, 342, 346, 351, 357, 361, 362, 364, 366, 383, 389, 394, 395, 397]}
GOAL_STATE_MAP = {"4x4": [15], "20x20": [399]}
cmap = 'cool'


def visualize_env(env, name):
    shape = env.desc.shape
    M = shape[0]
    N = shape[1]
    arr = np.zeros(shape)
    for i in range(M):
        for j in range(N):
            if (N * i + j) in TERM_STATE_MAP[name]:
                arr[i, j] = 0.25
            elif (N * i + j) in GOAL_STATE_MAP[name]:
                arr[i, j] = 1.0
    fig, ax = ply.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap=cmap)
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)
    for i in range(M):
        for j in range(N):
            if (i, j) == (0, 0):
                ply.text(j, i, 'S', ha='center', va='center', color='k', size=18)
            elif (N * i + j) in TERM_STATE_MAP[name]:
                ply.text(j, i, 'H', ha='center', va='center', color='k', size=18)
            elif (N * i + j) in GOAL_STATE_MAP[name]:
                ply.text(j, i, 'G', ha='center', va='center', color='k', size=18)
            else:
                ply.text(j, i, 'F', ha='center', va='center', color='k', size=18)
    fig.tight_layout()
    ply.savefig("figs/" + name + " map")


def value_iteration(env, descrip):
    V = np.zeros(env.nS, dtype='float64')  # initialize value-function
    max_iter = 20000
    theta = 1e-10
    time_arr = []
    nIter = []
    gamma_arr = np.linspace(0.5, 0.99, 12)
    for g in gamma_arr:
        value_diff = []
        t0 = time()
        for i in range(max_iter):
            prev_V = np.copy(V)
            for state in range(env.nS):
                A = next_step(env, state, V, gamma=g)
                V[state] = max(A)
            vD = np.sum(np.fabs(prev_V - V))
            value_diff.append(vD)
            if vD <= theta:
                tD = time() - t0
                print("gamma value is " + str(g))
                print('Value iteration converged at iteration# %d.' % (i + 1))
                print('Value iteration convergence took %.2fs' % (tD))
                break
        ply.plot(value_diff, label='gamma= %2.2g, iter= %3d' % (g, i + 1))
        time_arr.append(tD)
        nIter.append(i + 1)

    ply.title(descrip + ': Value Iteration Convergence Iteration-Gamma', fontsize=12, fontweight='bold')
    ply.ylabel('||v-v*||')
    ply.xlabel('# Iterations')
    ply.legend(loc='upper right')
    ply.ylim([0, 0.3])
    ply.grid()
    fig1 = ply.gcf()
    ply.draw()
    fig1.savefig('figs/VI_Convergence_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()

    tim = np.asarray(time_arr)
    ply.plot(gamma_arr, tim)
    ply.title(descrip + ': Value Iteration Convergence Time-Gamma', fontsize=12, fontweight='bold')
    ply.ylabel('Time to convergence(s)')
    ply.xlabel('Gamma (discount rate)')
    ply.grid()
    fig1 = ply.gcf()
    ply.draw()
    fig1.savefig('figs/VI_GammaTime_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()
    policy = np.zeros(env.nS, dtype='int16')
    for state in range(env.nS):
        A = next_step(env, state, V, gamma=0.99)
        policy[state] = np.argmax(A)
    return policy


def gmmma_rewards_VI(env, g):
    V = np.zeros(env.nS, dtype='float64')
    max_iter = 20000
    theta = 1e-10
    value_diff = []
    for i in range(max_iter):
        prev_V = np.copy(V)
        for state in range(env.nS):
            A = next_step(env, state, V, gamma=g)
            V[state] = max(A)
        vD = np.sum(np.fabs(prev_V - V))
        value_diff.append(vD)
        if vD <= theta:
            break
    policy = np.zeros(env.nS, dtype='int16')
    for state in range(env.nS):
        A = next_step(env, state, V, gamma=g)
        policy[state] = np.argmax(A)
    rewards = run_episodes(env, policy)
    print(f'Value iteration: average reward over 1000  episodes = {rewards}')



def next_step(env, state, V, gamma):
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + gamma * V[next_state])
    return A


def plot_policy(map, policy, algorithm):
    data = map_discretize(MAPS[map])
    np_pol = to_numpy(policy)
    ply.imshow(data, interpolation="nearest")
    for i in range(np_pol[0].size):
        for j in range(np_pol[0].size):
            arrow = '\u2190'
            if np_pol[i, j] == 1:
                arrow = '\u2193'
            elif np_pol[i, j] == 2:
                arrow = '\u2192'
            elif np_pol[i, j] == 3:
                arrow = '\u2191'
            text = ply.text(j, i, arrow,
                            ha="center", va="center", color="w")
    ply.savefig('figs/' + algorithm + '_optimal policy_FrozenLake' + map + '.png', bbox_inches='tight', dpi=200)
    ply.close()


def map_discretize(m):
    size = len(m)
    dis_map = np.zeros((size, size))
    for i, row in enumerate(m):
        for j, loc in enumerate(row):
            if loc == "S":
                dis_map[i, j] = 0
            elif loc == "F":
                dis_map[i, j] = 0
            elif loc == "H":
                dis_map[i, j] = -1
            elif loc == "G":
                dis_map[i, j] = 1
    return dis_map


def to_numpy(p):
    size = int(np.sqrt(len(p)))
    pol_num = np.asarray(p)
    pol_num = pol_num.reshape((size, size))
    return pol_num


def run_episodes(env, policy):
    numIter = 1000
    rewards = []
    numSteps = np.zeros(numIter, dtype='float32')
    numEpisodes = 10000
    env.reset()
    for iter in range(numIter):
        state = env.reset()
        step = 0
        done = False
        totR = 0;
        for step in range(numEpisodes):
            new_state, reward, done, _ = env.step(policy[state])
            totR += reward
            if done:
                break
            state = new_state
        numSteps[iter] = step
        rewards.append(totR)
    env.close()
    AvgRewards = np.mean(rewards)
    return AvgRewards


def view_episode(env, policy):
    obs = env.reset()
    step_idx = 0
    while True:
        env.render()
        obs, _, done, _ = env.step(int(policy[obs]))
        step_idx += 1
        if done:
            break
    return


def getRewardProb(env, sz):
    sz = int(sz ** 2)
    r = np.zeros((4, sz, sz))
    p = np.zeros((4, sz, sz))
    envP = env.unwrapped.P
    for state in envP:
        for action in envP[state]:
            transitions = envP[state][action]
            for t_idx in range((len(transitions))):
                new_state = transitions[t_idx][1]
                trans_prob = transitions[t_idx][0]
                reward = transitions[t_idx][2]
                p[action][state][new_state] += trans_prob
                r[action][state][new_state] += reward
            p[action, state, :] /= np.sum(p[action, state, :])
    return r, p


# Policy Iteration
def policy_iteration(env, descrip):
    policy = np.random.choice(env.nA, size=(env.nS))  # Random policies
    V = np.zeros(env.nS)
    idx = 1
    ix = 1
    theta = 1e-10
    timeV = []
    nVIter = []
    nPIter = []
    policy_arr = [[]]
    gamma_arr = np.linspace(0.5, 0.99, 12)
    for g in gamma_arr:
        t0 = time()
        while True:
            prev_V = np.copy(V)
            for state in range(env.nS):
                currPolicy = policy[state]
                V[state] = sum([p * (r + g * prev_V[s_]) for p, s_, r, _ in env.P[state][currPolicy]])
            if np.sum((np.fabs(prev_V - V))) <= theta:
                nVIter.append(idx)
                break
            idx += 1
        while True:
            prev_policy = np.copy(policy)
            policy = np.zeros(env.nS, dtype='int16')
            for state in range(env.nS):
                value = next_step(env, state, V, gamma=g)
                policy[state] = np.argmax(value)
            if np.all(prev_policy == policy):
                print("gamma value is " + str(g))
                print('Policy-iteration converged at iteration# %d.' % (ix))
                nPIter.append(ix)
                policy_arr.append(policy)
                break
            ix += 1
        timeV.append(time() - t0)
    tim = np.asarray(timeV)
    nItem = np.asarray(nPIter)
    ply.plot(gamma_arr, tim)
    ply.title(descrip + ': Policy Iteration Convergence Time-Gamma', fontsize=12, fontweight='bold')
    ply.ylabel('Time to convergence(s)')
    ply.xlabel('Gamma (discount rate)')
    ply.grid()
    fig1 = ply.gcf()
    ply.draw()
    fig1.savefig('figs/PI_GammaTime_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()

    ply.plot(gamma_arr, nItem, color="blue", marker='o')
    ply.title(descrip + ': Policy Iteration Convergence Iteration-Gamma', fontsize=12, fontweight='bold')
    ply.xlabel('Discount rate(gamma)')
    ply.ylabel('# Iterations')
    ply.grid()
    fig1 = ply.gcf()
    ply.draw()
    fig1.savefig('figs/IterNum_PI_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()
    for i in range(1,13):
        rewards = run_episodes(env, policy_arr[i])
        print(f'Policy iteration: average reward over 1000  episodes = {rewards}')
    return policy_arr[-1]


def QLearningRL(env, descrip,lrate, gamma=0.99):
    max_steps = 1000000
    Qtbl = np.zeros([env.nS, env.nA])
    num_episodes = 50000
    lr = lrate
    epsi = []
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.1  # Minimum exploration probability
    dr = 0.005  # Exponential decay rate for exploration prob
    rewards = []  # np.zeros(num_episodes) # List of rewards
    epsilon = 1.0
    env = env.unwrapped
    t0 = time()
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        totalR = 0
        epsi.append(epsilon)
        for step in range(max_steps):
            a = np.argmax(Qtbl[state, :] + np.random.randn(1, env.nA) * (1. / (episode + 1)))
            nextSt, r, done, _ = env.step(a)  # new state and reward
            Qtbl[state, a] += lr * ((r + gamma * np.max(Qtbl[nextSt, :])) - Qtbl[state, a])
            state = nextSt
            totalR += r
            if done == True:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-dr * episode)
        rewards.append(totalR)
    print (time()-t0)
    optimalPolicy = np.argmax(Qtbl, axis=1)
    rolling_reward = np.split(np.array(rewards), num_episodes / 500)
    count = 500
    print("*****Average reward per thousand episode*****\n")
    for rr in rolling_reward:
        print(count, ": ", str(sum(rr / 500)))
        count += 500
    tmp = pd.DataFrame(data=rewards)
    ply.plot(range(num_episodes - 49), tmp.rolling(50).mean().dropna())
    ply.plot(range(num_episodes), tmp.rolling(500).mean())
    ply.title(descrip + ': QL MeanRewards'+"(learning rate="+str(lrate)+")", fontsize=12, fontweight='bold')
    ply.xlabel('# of episodes')
    ply.ylabel('Rolling Mean Reward')
    fig1 = ply.gcf()
    ply.draw()
    fig1.savefig('figs/QL_RollingMeanReward_' + descrip +"_"+str(lrate)+ '.png', bbox_inches='tight', dpi=200)
    ply.close()
    return optimalPolicy


def displayResults(env, optimalPolicy):
    env.reset()
    for episode in range(5):
        state = env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(env.spec.max_episode_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            new_state, reward, done, _ = env.step(optimalPolicy[state])
            if done:
                env.render()  # print the last state
                print("Number of steps", step)  # number of steps it took.
                break
            state = new_state
    env.close()


if __name__ == '__main__':
    print("----------------------------- Running FrozenLake4x4 value iteration -----------------------------")
    value_iteration(frozenLake4x4, 'FrozenLake4x4')
    plot_policy("4x4", value_iteration(frozenLake4x4, 'FrozenLake4x4'), 'VI')
    gmmma_rewards_VI(frozenLake4x4, 0.5)
    gmmma_rewards_VI(frozenLake4x4,0.54)
    gmmma_rewards_VI(frozenLake4x4, 0.59)
    gmmma_rewards_VI(frozenLake4x4, 0.63)
    gmmma_rewards_VI(frozenLake4x4, 0.68)
    gmmma_rewards_VI(frozenLake4x4, 0.72)
    gmmma_rewards_VI(frozenLake4x4, 0.77)
    gmmma_rewards_VI(frozenLake4x4, 0.81)
    gmmma_rewards_VI(frozenLake4x4, 0.86)
    gmmma_rewards_VI(frozenLake4x4, 0.9)
    gmmma_rewards_VI(frozenLake4x4, 0.95)
    gmmma_rewards_VI(frozenLake4x4, 0.99)

    print("----------------------------- Running FrozenLake4x4 policy iteration -----------------------------")
    policy_iteration(frozenLake4x4, 'FrozenLake4x4')
    plot_policy("4x4", policy_iteration(frozenLake4x4, 'FrozenLake4x4'), 'PI')

    print("----------------------------- Running FrozenLake4x4 Q learning -----------------------------")
    QLearningRL(frozenLake4x4, 'FrozenLake4x4',0.1)
    QLearningRL(frozenLake4x4, 'FrozenLake4x4', 0.75)
    plot_policy("4x4", QLearningRL(frozenLake4x4, 'FrozenLake4x4',0.75), 'QL')

    print("----------------------------- Running FrozenLake20x20 value iteration -----------------------------")
    value_iteration(frozenLake20x20, 'FrozenLake20x20')
    plot_policy("20x20", value_iteration(frozenLake20x20, 'FrozenLake20x20'), 'VI')
    gmmma_rewards_VI(frozenLake20x20, 0.5)
    gmmma_rewards_VI(frozenLake20x20, 0.54)
    gmmma_rewards_VI(frozenLake20x20, 0.59)
    gmmma_rewards_VI(frozenLake20x20, 0.63)
    gmmma_rewards_VI(frozenLake20x20, 0.68)
    gmmma_rewards_VI(frozenLake20x20, 0.72)
    gmmma_rewards_VI(frozenLake20x20, 0.77)
    gmmma_rewards_VI(frozenLake20x20, 0.81)
    gmmma_rewards_VI(frozenLake20x20, 0.86)
    gmmma_rewards_VI(frozenLake20x20, 0.9)
    gmmma_rewards_VI(frozenLake20x20, 0.95)
    gmmma_rewards_VI(frozenLake20x20, 0.99)

    print("----------------------------- Running FrozenLake20x20 policy iteration -----------------------------")
    policy_iteration(frozenLake20x20, 'FrozenLake20x20')
    plot_policy("20x20", policy_iteration(frozenLake20x20, 'FrozenLake20x20'), 'PI')

    print("----------------------------- Running FrozenLake20x20 Q learning -----------------------------")
    QLearningRL(frozenLake20x20, 'FrozenLake20x20',0.1)
    QLearningRL(frozenLake20x20, 'FrozenLake20x20', 0.75)
    plot_policy("20x20", QLearningRL(frozenLake20x20, 'FrozenLake20x20',0.75), 'QL')
    visualize_env(frozenLake20x20, "20x20")

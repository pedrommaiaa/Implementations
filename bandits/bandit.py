import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

matplotlib.use('Agg')
np.random.seed(10)


class Bandit:
    def __init__(self, n_arm=10, epsilon=[0.], step_size=0.1):
        self.n = n_arm
        self.step_size = step_size
        self.indices = np.arange(self.n)
        self.epsilons = [x for x in epsilon]

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.n)

        # estimation for each action
        self.q_estimation = np.zeros(self.n)

        # # of chosen times for each action
        self.action_count = np.zeros(self.n)
        
        self.best_action = np.argmax(self.q_true)


    # get an action for this bandit
    def act(self, epsilon):
        if np.random.uniform(0,1) < epsilon:
            return np.random.choice(self.indices)
        else:
            q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.action_count[action] += 1

        # update estimation using sample averages
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        return reward


    def simulate(self, epochs=2000, times=1000):
        rewards = np.zeros((len(self.epsilons), epochs, time))
        best_action_counts = np.zeros(rewards.shape)
        for i, epsilon in enumerate(self.epsilons):
            for r in trange(epochs):
                self.reset()
                for t in range(time):
                    action = bandit.act(epsilon)
                    reward = bandit.step(action)
                    rewards[i, r, t] = reward
                    if action == bandit.best_action:
                        best_action_counts[i, r, t] = 1
        mean_best_action_counts = best_action_counts.mean(axis=1)
        mean_rewards = rewards.mean(axis=1)
        return mean_best_action_counts, mean_rewards


if __name__ == "__main__":
    
    epsilons = [0, 0.01, 0.1] 
    bandit = Bandit(epsilon=epsilons)
    best_action_counts, rewards = bandit.simulate()
    
    plt.figure()
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()
    plt.savefig('average_reward.png')
    plt.close

    plt.figure()
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.savefig('optimal_action.png')
    plt.close

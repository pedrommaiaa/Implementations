import gym
import math
import numpy as np
from collections import deque



class QLearning(object):
    """
    Template for the Q-Learning algorithm.
    
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    """
    def __init__(self, environment, n_episodes=1000, gamma=0.98, min_epsilon=0, min_alpha=0.1, render=False):
        
        self.env = environment
        self.render = render
        self.n_episodes = n_episodes

        self.buckets=(1,1,6,12,)
        
        self.gamma = gamma
        
        self.epsilon = 1.0 # Probability of choosing a random action
        self.epsilon_decay = 0.98 # decay of epsilon per episode
        self.min_epsilon = min_epsilon

        self.alpha = 1
        self.alpha_decay = 0.98
        self.min_alpha = min_alpha

        # Action-value function
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    
    def learn(self):
        scores = deque(maxlen=100)
        learned = False
        if self.render: display = 0

        for e in range(self.n_episodes+1):
            
            current_state = self.discretize(self.env.reset())
            i = 0
                
            while True:
                if self.render:
                    if display >= 1:
                        self.env.render()
                action = self.choose_action(current_state, self.epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                # update Q
                self.Q[current_state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[new_state]) - self.Q[current_state][action])
                 
                if done:
                    break
                current_state = new_state
                i += 1

            
            # decay exploration probability
            # Decaying epsilon over time allow us to explore random actions in the
            # beginning, but overtime we start to act more greedy and exploit the
            # actions that we aready know will give us the most reward for that 
            # particular state.
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)

            # learning rate decay
            # At the beginning, you want higher alpha so that you can take big steps
            # and learn things but while the agent is learning you should decay alpha
            # to stabilize the model output and eventually converge to an optimal policy.
            self.alpha *= self.alpha_decay
            self.alpha = max(self.alpha, self.min_alpha)

            scores.append(i)
            mean_score = np.mean(scores)
            
            if e == self.n_episodes-1:
                display+=1
            if mean_score >= 195 and e >= 100:
                learned = True 
                print(f'Ran {e} episodes. Solved after {e-100} trials ✔')
                if self.render:
                    if display >= 1: 
                        break
                    # Run one more time but tis time with display enabled
                    display += 1
                else:
                    break
            if e % 100 == 0:
                print(f'[Episode {e}] - Mean survival time over last 100 episodes was {mean_score} ticks.')

        if not learned:
            print(f'Did not solve after {e} episodes 😞')

        self.env.close()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    QLearning = QLearning(env, n_episodes=1000, render=True)
    QLearning.learn()

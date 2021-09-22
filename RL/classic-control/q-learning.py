import sys
import gym
import numpy as np
import math
from collections import deque

"""
The pendulum starts upright, and the goal is to prevent it from falling
over by increasing and reducing the cart's velocity

Observation: 
    [cart position, cart velocity, pole angle, pole angular velocity].

Actions:
    nActions: 2
    0 - Push cart to the left
    1 - Push cart to the right

Reward:
    Reward is 1 for every step taken, including the terminal step

Solved Requirements:
    Considered solved when the average return is >= 195.0 over 100 consecutive 
    trials.
"""

"""
Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
while following an epsilon-greedy policy.

Args:
    env: CartPole-v1 OpenAI gym environment.
        nActions = 2
    num_episodes: Number of episodes to sample.
    discount_factor: Gamma discount factor.
    epsilon: Chance of sampling a random action.
    alpha: TD learning rate.

Returns:
    Q: The optimal action-value function. A dictionary that maps from state -> value.
"""
def QCartPoleSolver(buckets=(1, 1, 6, 12,), n_episodes=1000, n_win_ticks=195, min_alpha=0.1, min_epsilon=0.1, gamma=1.0, ada_divisor=25):

    env = gym.make('CartPole-v1')

    Q = np.zeros(buckets + (env.action_space.n,))

    def discretize(obs):
        upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
        lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(state, epsilon):
        return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])

    def get_epsilon(t):
        return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))

    def get_alpha(t):
        return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))

    scores = deque(maxlen=100)
    learned = False
    test = 0

    for e in range(n_episodes):
        current_state = discretize(env.reset())

        alpha = get_alpha(e)
        epsilon = get_epsilon(e)
        i = 0
            
        while True:
            if test >= 1:
                env.render()
            action = choose_action(current_state, epsilon)
            obs, reward, done, _ = env.step(action)
            new_state = discretize(obs)
            # update Q
            Q[current_state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[current_state][action])
             
            if done:
                break
            current_state = new_state
            i += 1


        scores.append(i)
        mean_score = np.mean(scores)
        if mean_score >= n_win_ticks and e >= 100:
            learned = True 
            print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
            if test >= 1: 
                break
            test += 1
        if e % 100 == 0:
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

    if not learned:
        print('Did not solve after {} episodes 😞'.format(e))

    env.close()

if __name__ == "__main__":
    
    QCartPoleSolver()

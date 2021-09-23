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

def QCartPoleSolver(buckets=(1, 1, 6, 12,), n_episodes=1000, n_win_ticks=195, gamma=0.995, display_when_done=True):
    """
    Q-Learning algorithm for the CartPole Environment. Finds the optimal greedy 
    policy while following an epsilon-greedy policy.

    Args:
        buckets:
        n_episodes: Number of episodes to sample.
        n_win_ticks: Average return necessery to consider the problem solved.
        gamma: Gamma discount factor.
        display_when_done: Render env after completing solving requirements.

    Returns:
        Q: The optimal action-value function. 
    """
    epsilon = 1.0 # probability of choosing a random action
    epsilon_decay = 0.98 # decay of epsilon per episode
    min_epsilon = 0

    alpha = 1
    alpha_decay = 0.98
    min_alpha = 0.1

    # CartPole-v1 OpenAI gym environment.
    env = gym.make('CartPole-v1')

    # Action-value function
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


    scores = deque(maxlen=100)
    learned = False
    if display_when_done: display = 0

    for e in range(n_episodes):
        
        current_state = discretize(env.reset())
        i = 0
            
        while True:
            if display_when_done:
                if display >= 1:
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

        
        # decay exploration probability
        # Decaying epsilon over time allow us to explore random actions in the
        # beginning, but overtime we start to act more greedy and exploit the
        # actions that we aready know will give us the most reward for that 
        # particular state.
        epsilon *= epsilon_decay
        epsilon = max(epsilon, min_epsilon)

        # learning rate decay
        # At the beginning, you want higher alpha so that you can take big steps
        # and learn things but while the agent is learning you should decay alpha
        # to stabilize the model output and eventually converge to an optimal policy.
        alpha *= alpha_decay
        alpha = max(alpha, min_alpha)

        scores.append(i)
        mean_score = np.mean(scores)
        if mean_score >= n_win_ticks and e >= 100:
            learned = True 
            print(f'Ran {e} episodes. Solved after {e-100} trials ✔')
            if display_when_done:
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

    env.close()
    return Q

if __name__ == "__main__":
    
    Q = QCartPoleSolver()

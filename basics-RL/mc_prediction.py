import numpy as np
import sys
from gridWorld.grid import gridWorld
from collections import defaultdict


def mc_prediction(policy, env, num_episodes, discount_factor=0.99):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # The final value function
    V = np.zeros(env.nS)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        G = 0
        state = env.reset()
        # Look at each possible next action
        for action, _ in enumerate(policy[state]):
            for _, next_state, reward, done in env.P[state][action]:
                episode.append((state, reward))
                if done:
                    break
                state = next_state
                   

        for idx, step in enumerate(episode[::-1]):
            G = discount_factor*G + step[1]
            if step[0] not in np.array(episode[::-1])[:, 0][idx+1:]:
                returns_sum[step[0]] += G
                returns_count[step[0]] += 1.0
                #Returns[step[0]].append(G)
                V[step[0]] = returns_sum[step[0]] / returns_count[step[0]]

    print()
    return np.round(V, 1)


if __name__ == "__main__":

    env = gridWorld()
    random_policy = np.ones([env.nS, env.nA]) / env.nA

    V = mc_prediction(random_policy, env, num_episodes=500000).reshape(env.shape)
    print(V) 

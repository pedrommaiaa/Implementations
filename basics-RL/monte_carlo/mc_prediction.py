import sys
import numpy as np
from collections import defaultdict

sys.path.append('../')
np.random.seed(10)
from env.gridWorld import gridWorld




def mc_prediction(policy, env, num_episodes, discount=1.0):
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
        state = env.reset()
        while True:

            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        
        states_in_episode = set([(x[0]) for x in episode])
        for state in states_in_episode:
            first_occurence = next(i for i,x in enumerate(episode) if x[0] == state)
            G = sum([x[2]*(discount**i) for i, x in enumerate(episode[first_occurence:])])
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state] 
    
    print()
    return np.round(V)


if __name__ == "__main__":

    env = gridWorld()
    def random_policy(state):
        """
        A policy that selects random actions
        """
        policy = np.ones([env.nS, env.nA]) / env.nA
        actions = policy[state]
        return np.random.randint(0, len(actions))
    
    V = mc_prediction(random_policy, env, num_episodes=10000).reshape(env.shape)
    print(f"Value function:\n{V}") 

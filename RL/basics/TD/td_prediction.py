import sys
import numpy as np
from collections import defaultdict

if "../../" not in sys.path:
    sys.path.append('../../')
np.random.seed(10)
from env.gridWorld import gridWorld



def td_prediction(policy, env, num_episodes, discount=1.0, alpha=0.01):
    """
    Tabular TD(0) algorithm. Calculates the value function for a given policy.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        alpha: learning_rate 
    
    Returns:
        V: Final value function
    """
    # The final value function
    V = np.zeros(env.nS)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print(f"\rEpisode {i_episode}/{num_episodes}.", end="")
            sys.stdout.flush()


        state = env.reset()
        while True:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            # updating
            V[state] += alpha*(reward + discount*V[next_state] - V[state]) 
            if done:
                break
            state = next_state
         
    print()
    return np.round(V)


if __name__ == "__main__":

    env = gridWorld()
    def random_policy(state):
        """
        A policy that selects random actions
        """
        policy = np.ones([env.nS, env.nA]) / env.nA
        return np.random.randint(0, len(policy[state]))
    
    V = td_prediction(random_policy, env, num_episodes=10000).reshape(env.shape)
    print(f"Value function:\n{V}") 

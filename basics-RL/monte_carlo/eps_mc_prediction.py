import sys
import numpy as np
from collections import defaultdict

sys.path.append('../')
np.random.seed(10)
from env.gridWorld import gridWorld


def mc_prediction(env, num_episodes, discount=1.0, eps=0.1):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy.
    
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
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    #Q = np.ones([env.nS, env.nA]) / env.nA
    Q = defaultdict(lambda: np.zeros(env.nA)) 
    
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    def policy_fn(state, policy):
        A = np.ones(env.nA, dtype=float) * eps / env.nA
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - eps)
        policy[state] = np.eye(env.nA)[best_action]
        return A
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print(f"\rEpisode {i_episode}/{num_episodes}.", end="")
            sys.stdout.flush()


        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        while True:
            probs = policy_fn(state, policy)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        
        states_in_episode = set([(x[0], x[1]) for x in episode]) # unique states visited
        for state, action in states_in_episode:
            sa_pair = (state, action)
            # for each unique state, get the index in episode of the first occurence of that state
            first_occurence = next(i for i,x in enumerate(episode) 
                                   if x[0] == state and x[1] == action)
            # sum all the rewards*gamma for each first occurence state in episode 
            G = sum([x[2]*(discount**i) for i, x in enumerate(episode[first_occurence:])])
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair] 
         
    print()
    return Q, policy


if __name__ == "__main__":

    env = gridWorld()
    
    Q, policy = mc_prediction(env, 500000)
    
    V = np.zeros(env.nS)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value

    print(f"Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):\n{np.reshape(np.argmax(policy, axis=1), env.shape)}\n")

    print(f"Reshaped Grid Value Function:\n{np.round(V.reshape(env.shape))}\n")

import sys
import numpy as np
from collections import defaultdict

if "../" not in sys.path:
    sys.path.append('../')
np.random.seed(10)
from env.gridWorld import gridWorld


def create_target_policy_fn():
    def target_policy_fn(A):
        epsilon = 0.25
        idx = np.argmax(A)
        probs = []
        A_ = np.sqrt(sum([i**2 for i in A]))
        if A_ == 0:
            A_ = 1.0
        for i, a in enumerate(A):
            if i == idx:
                probs.append(round(1-epsilon + (epsilon/A_),3))
            else:
                probs.append(round(epsilon/A_,3))
        return np.array(probs)
    return target_policy_fn



def sampling_mc_prediction(env, num_episodes, discount=0.99):
    """
    Monte Carlo Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        Q: Action-value function.
    """
    # The final action-value function
    Q = defaultdict(lambda: np.zeros(env.nA)) 
    # The cumulative denominator of the weighted importance (across all episodes)
    C = np.zeros((env.nS, env.nA))

    # Behavior policy - random policy actions and probabilities for each action
    def behavior_policy(state):
        A = np.ones([env.nS, env.nA], dtype=float) / env.nA
        return A[state]

    # Target Policy 
    target_policy_fn = create_target_policy_fn()

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
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio(the weights of the returns)
        W = 1.0
        
        for state, action, reward in reversed(episode):
            G = discount*G + reward
            # Update weighted importance sampling formula denominator
            C[state][action] += W
            # Update the action-value function using the incremental update formula
            # This also improves our target policy which holds a referece to V
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            
            target_policy = target_policy_fn(Q[state])

            W = W * target_policy[action]/behavior_policy(state)[action]
            if W == 0:
                break 
        

    print()
    return Q


if __name__ == "__main__":

    env = gridWorld()
    Q = sampling_mc_prediction(env, 10000) 
    
    V = np.zeros(env.nS)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    
    print(f"Value function:\n{np.round(V.reshape(env.shape))}\n")

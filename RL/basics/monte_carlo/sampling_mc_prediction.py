import sys
import numpy as np
from collections import defaultdict

if "../../" not in sys.path:
    sys.path.append('../../')
np.random.seed(10)
from env.gridWorld import gridWorld

def action_probs(A):
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
    err = sum(probs)-1

    return np.array(probs)


def sampling_mc_prediction(env, num_episodes, discount=0.99):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each aciton.
        discount_factor: Gamma discount factor.
    
    Returns:
    """
    # The final action-value function
    Q = defaultdict(lambda: np.zeros(env.nA)) 
    # The cumulative denominator of the weighted importance (across all episodes)
    C = np.zeros((env.nS, env.nA))

    # Behavior policy - random policy actions and probabilities for each action
    def behavior_policy_fn():
        A = np.ones(env.nA, dtype=float) / env.nA
        return A

    # target policy - in this case I used the soft-epsilon policy of the on-policy MC
    target_policy = np.ones([env.nS, env.nA]) / env.nA 
    def target_policy_fn(state, target_policy):
        A = np.ones(env.nA, dtype=float) * discount / env.nA
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - discount)
        target_policy[state] = np.eye(env.nA)[best_action]
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
            probs = behavior_policy_fn()
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
            
            target_policy_fn(state, target_policy)

            action_values = Q[state]
            probs = action_probs(action_values)
            
            W = W * probs[action]/behavior_policy_fn()[action]
            if W == 0:
                break 
        

    print()
    return Q, target_policy


if __name__ == "__main__":

    env = gridWorld()
    Q, target_policy = sampling_mc_prediction(env, 50000) 
    
    V = np.zeros(env.nS)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    
    print(f"Grid Policy (0=up, 1=right, 2=down, 3=left):\n{np.argmax(target_policy, axis=1).reshape(env.shape)}\n")

    print(f"Value function:\n{np.round(V.reshape(env.shape))}\n")

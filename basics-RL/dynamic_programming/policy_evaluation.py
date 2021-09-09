import sys
import numpy as np

sys.path.append('../')
np.random.seed(10)
from env.gridWorld import gridWorld

# Problem: Evaluate a given policy pi
# Solution: Iterative application of Bellman expectation backup

def policy_evaluation(policy, env, discount_factor=1.0, threshold=0.00001):
    """
    Evaluate a policy given an environment anda full description of the 
    environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
             env.nS is the number of states in the environment.
             env.nA is the number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor

    Returns:
        Vector of legnth env.nS representing the value function.
    """

    # Start with a all 0 value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # for each state perform a full backup
        for state in range(env.nS):
            v = 0
            # Look at each possiple next action
            for action, action_prob in enumerate(policy[state]):
                # For each action, look at the possible next state
                for prob, next_state, reward, done in env.P[state][action]:
                    # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        if delta < threshold:
            break
    return np.round(np.array(V), 1)


if __name__ == "__main__":

    env = gridWorld()
    #env.display()
    
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    
    V = policy_evaluation(random_policy, env)

    print(f"Value Function:\n{V.reshape(env.shape)}")

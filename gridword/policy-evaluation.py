import numpy as np
from grid import Gridworld

# Problem: Evaluate a given policy pi
# Solution: Iterative application of Bellman expectation backup

def policy_evaluation(policy, env):

    """
    Evaluate a policy given an environment anda full description of the 
    environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: Grid World.
             env.nStates is the number of states in the environment.
             env.nActions is the number of actions in the environment.
    """

    # Start with a all 0 value function
    V = np.zeros(env.nStates)
    while True:
        # Implement!
        break
    return np.array(V)



if __name__ == "__main__":

    env = Gridworld()
    
    random_policy = np.ones([env.nStates, env.nActions]) / env.nActions
    
    v = policy_evaluation(random_policy, env)

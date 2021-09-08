import numpy as np
from gridWorld.grid import gridWorld


# From policy-evaluation!
def policy_evaluation(policy, env, discount_factor=1.0, theta=0.00001):
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
        if delta < theta:
            break
    return np.array(V)



def policy_improvement(env, policy_eval_fn=policy_evaluation, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI env.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state
        s contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
    """
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all actions in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # ties are resolved arbitrary
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


if __name__ == "__main__":
    
    env = gridWorld()

    policy, v = policy_improvement(env)
     
    env.display()
    print()

    print(f"Policy Probability Distribution:\n {policy}\n")

    print(f"Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):\n {np.reshape(np.argmax(policy, axis=1), env.shape)}\n")

    print(f"Value Function:\n {v}\n")

    print(f"Reshaped Grid Value Function:\n {v.reshape(env.shape)}\n")

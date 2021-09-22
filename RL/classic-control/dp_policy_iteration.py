import sys
import numpy as np

if "../" not in sys.path:
    sys.path.append('../')
np.random.seed(10)
from env.gridWorld import gridWorld
from dp_policy_evaluation import policy_evaluation


def policy_iteration(env, policy_eval_fn=policy_evaluation, discount_factor=1.0):
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
    # 1. Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        
        # 2. Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        # 3. Policy Improvement
        policy_stable = True
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # ties are resolved arbitrary
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
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

    policy, v = policy_iteration(env)

    print(f"Policy Probability Distribution:\n{policy}\n")

    print(f"Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):\n{np.reshape(np.argmax(policy, axis=1), env.shape)}\n")

    print(f"Reshaped Grid Value Function:\n{v.reshape(env.shape)}\n")

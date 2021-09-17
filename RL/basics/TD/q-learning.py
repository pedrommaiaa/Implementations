import sys
import numpy as np
from collections import defaultdict

if "../../" not in sys.path:
    sys.path.append('../../')
np.random.seed(10)
from env.gridWorld import gridWorld


def q_learning(env, num_episodes, discount=0.99, epsilon=0.1, alpha=0.01):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance of sampling a random action.
        alpha: TD learning rate.
 
    Returns:
        Q: The optimal action-value function. A dictionary that maps from state -> value.
        Policy: Epsilon-greedy policy. A numpy array with length (env.nS, env.nA).
    """ 
    # The final action-value function.
    Q = defaultdict(lambda: np.zeros(env.nA))    
    
    final_policy = np.ones([env.nS, env.nA]) / env.nA
    
    # The policy we're following
    def policy(state, final_policy):
        """ e-greedy policy """
        if np.random.random() > epsilon:
            best_action = np.argmax(Q[state])
            final_policy[state] = np.eye(env.nA)[best_action]
            return best_action
        else:
            return np.random.randint(0, len(Q[state]))

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print(f"\rEpisode {i_episode}/{num_episodes}.", end="")
            sys.stdout.flush()

        state = env.reset()
        while True:
            action = policy(state, final_policy)
            next_state, reward, done, _ = env.step(action)
            
            best_next_action = np.argmax(Q[next_state])

            Q[state][action] += alpha*(reward + discount*Q[next_state][best_next_action] - Q[state][action])

            if done:
                break
            state = next_state

         
    print()
    return Q, final_policy


if __name__ == "__main__":

    env = gridWorld()
    
    Q, policy = q_learning(env, 10000)
    
    V = np.zeros(env.nS)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value

    print(f"Grid Policy (0=up, 1=right, 2=down, 3=left):\n{np.reshape(np.argmax(policy, axis=1), env.shape)}\n")
    
    print(f"Grid Value Function:\n{np.round(V.reshape(env.shape))}\n")

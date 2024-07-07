from Q_Learning_LunarLander import *
import gymnasium as gym

# Load the lunar lander environment
env = gym.make('LunarLander-v2', render_mode="human")

# Dimensions of the state space and action space
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Training hyperparameters
num_episodes = 1000
replay_buffer = ReplayBuffer(10000)

# Model hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64

agent = DQN_Agent(state_dim, action_dim, replay_buffer, gamma, epsilon, epsilon_decay,
                  epsilon_min, batch_size)

# Train the agent
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        experience = (state, action, reward, next_state, done)
        replay_buffer.add(experience)
        state = next_state
        total_reward += reward

        agent.train()

        if done:
            break

    agent.update_target_network()

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")



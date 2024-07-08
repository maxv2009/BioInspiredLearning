import gymnasium as gym
from plot_creation import *

# Choose between tensorflow and pytorch
chosen_framework = "pytorch"

# Training hyperparameters
num_episodes = 2000

# Model hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 128
learning_rate = 0.001
buffer_size = 200000

# Load the lunar lander environment
env = gym.make('LunarLander-v2', render_mode="human")

# Dimensions of the state space and action space
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

if chosen_framework == "tensorflow":
    from DQN_LunarLander_TensorFlow import *
    print("Using TensorFlow")
else:
    from DQN_LunarLander_PyTorch import *
    print("Using PyTorch")

# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size)

# Initialize the DQN agent
agent = DQN_Agent(state_dim, action_dim, replay_buffer, gamma, epsilon, epsilon_decay,
                  epsilon_min, batch_size, learning_rate)

reward_progress = []
episodes = []

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

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    reward_progress.append(total_reward)
    episodes.append(episode)

    plot_rewards(episodes, reward_progress)


    # Save the model every 25 episodes
    if (episode + 1) % 25 == 0:
        agent.save_model(f".\\save\\run_2_dqn_lunarlander_episode_{episode + 1}.h5")

# Save plot
plot_save()

# Save the final model
agent.save_model("dqn_lunarlander_final.h5")





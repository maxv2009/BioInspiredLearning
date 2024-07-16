import gymnasium as gym
from plot_creation import *
from pathlib import Path
import os

# Choose whether you want to load an existing agent or train one
train_or_play = "train"

# Choose between tensorflow and pytorch
chosen_framework = "pytorch"

# Path to the saved agent in case an existing agent is to be used
model_path = "epsilon_decay_0.999_epsilon_min_0.005_learning_rate_0.001/dqn_lunarlander_final.h5"

# Training hyperparameters
num_episodes = 1000

# Model hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay_list = [0.999]
epsilon_min_list = [0.01, 0.005]
batch_size = 64
learning_rate_list = [0.0005, 0.001]
buffer_size = 1000000

# Load the lunar lander environment with or without rendering
if train_or_play == "train":
    env = gym.make('LunarLander-v2')
else:
    env = gym.make('LunarLander-v2', render_mode='human')

# Dimensions of the state space and action space
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Choose between TensorFlow or PyTorch implementation
if chosen_framework == "tensorflow":
    from DQN_LunarLander_TensorFlow import *
    print("Using TensorFlow")
else:
    from DQN_LunarLander_PyTorch import *
    print("Using PyTorch")

# Lists to store rewards and episodes for plotting
all_rewards = []
all_episodes = []
all_names = []

# Train
if train_or_play == "train":
    for learning_rate in learning_rate_list:
        for epsilon_min in epsilon_min_list:
            for epsilon_decay in epsilon_decay_list:
                for repeat in range(5):
                    # Create a replay buffer
                    replay_buffer = ReplayBuffer(buffer_size)

                    foldername = (f".\\repeatedrun\\epsilon_decay_{epsilon_decay}_epsilon_min"
                                    f"_{epsilon_min}_learning_rate_{learning_rate}_run{repeat}")
                    Path(os.getcwd(), foldername).mkdir(parents=True, exist_ok=True)

                    # Initialize the DQN agent
                    agent = DQN_Agent(state_dim, action_dim, replay_buffer, gamma, epsilon, epsilon_decay,
                                      epsilon_min, batch_size, learning_rate)

                    reward_progress = []
                    episodes = []

                    # Train the agent
                    for episode in range(num_episodes):
                        state, _ = env.reset()
                        total_reward = 0

                        # Run an episode with up to 1000 timesteps. After 1000 timesteps the episode is aborted.
                        for t in range(1000):
                            # Select an action (epsilon-greedy policy)
                            action = agent.select_action(state)
                            # Take a step
                            next_state, reward, done, _, _ = env.step(action)
                            # Add the experience to the replay buffer
                            experience = (state, action, reward, next_state, done)
                            replay_buffer.add(experience)
                            # Update the current state to the next state
                            state = next_state
                            # Update the total reward
                            total_reward += reward

                            # Train the agent
                            agent.train()

                            if done:
                                break

                        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

                        # Save the rewards for plotting
                        reward_progress.append(total_reward)
                        episodes.append(episode)

                        # Plot results every 25 episodes
                        if (episode + 1) % 25 == 0:
                            plot_rewards(episodes, reward_progress, train_or_play, gamma, epsilon,
                                         epsilon_decay, epsilon_min,
                                     batch_size, learning_rate, buffer_size)

                        # Save model every 100 episodes
                        if (episode + 1) % 100 == 0:
                            agent.save_model(f".\\{foldername}\\dqn_lunarlander_episode_{episode + 1}.h5")

                        if not len(reward_progress) < 200:
                            last_200_entries = reward_progress[-200:]
                            average = sum(last_200_entries) / 200
                            if average >= 240:
                                print(f"Solved in {episode + 1} episodes!")
                                break

                    all_rewards.append(reward_progress)
                    all_episodes.append(episodes)
                    all_names.append("Epsilon Decay: " + str(epsilon_decay) + " Epsilon min.: " + str(
                        epsilon_min) + " Learning Rate: " + str(learning_rate) + " Run: " + str(
                        repeat))

                    # Save plot
                    plot_rewards(episodes, reward_progress, train_or_play, epsilon, epsilon_decay,
                                 epsilon_min,
                                 batch_size, learning_rate, buffer_size, save=True,
                                 foldername=foldername)

                    # Save the final model
                    agent.save_model(f".\\{foldername}\\dqn_lunarlander_final_episode_{episode}.h5")

                foldername = "repeatedrun"
                Path(os.getcwd(), foldername).mkdir(parents=True, exist_ok=True)

                # Plot all results
                plot_multiple_rewards(all_rewards, all_episodes, all_names, foldername, learning_rate)
                save_lists(f".\\{foldername}\\all_lists_lr_{learning_rate}.pkl", (all_rewards,
                                                                                  all_episodes, all_names))
                all_rewards = []
                all_episodes = []
                all_names = []


else:
    # Create a replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Set epsilon to zero so no exploration is performed
    epsilon = 0
    agent = DQN_Agent(state_dim, action_dim, replay_buffer, gamma, epsilon,  epsilon_decay_list[
        0], epsilon_min_list[0], batch_size, learning_rate_list[0])
    agent.load_model(f".\\{model_path}")
    reward_progress = []
    episodes = []

    # Train the agent
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        # Run an episode with up to 1000 timesteps. After 1000 timesteps the episode is aborted.
        for t in range(1000):
            # Select an action (epsilon-greedy policy)
            action = agent.select_action(state)
            # Take a step
            next_state, reward, done, _, _ = env.step(action)
            # Update the current state to the next state
            state = next_state
            # Update the total reward
            total_reward += reward

            if done:
                break

        # Save the rewards for plotting
        reward_progress.append(total_reward)
        episodes.append(episode)

        # Plot results every 5 episodes
        if (episode + 1) % 5 == 0:
            plot_rewards(episodes, reward_progress, train_or_play)

    plot_rewards(episodes, reward_progress, train_or_play, save=True, foldername="play")

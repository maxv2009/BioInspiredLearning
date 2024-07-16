import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# Create the model
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Class for the replay buffer. The replay buffer stores experiences and samples a batch of
    experiences to train the agent."""
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    # Add experience to the buffer
    def add(self, experience):
        self.buffer.append(experience)

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size):
        return random.sample(self.buffer, int(batch_size))

    # Get the size of the buffer
    def __len__(self):
        return len(self.buffer)


class DQN_Agent:
    """Class for the DQN agent. The agent selects actions, trains the Q-network, and updates the
    target network."""
    def __init__(self, state_dim, action_dim, replay_buffer, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, learning_rate=0.001):
        # Get the dimensions of the state and action space
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Get the replay buffer, gamma, epsilon, epsilon decay, epsilon min, batch size,
        # and learning rate
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Update the target network every 10 steps
        self.target_update = 10
        self.step_count = 0
        # Check if a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        # Set the target network weights to the Q-network weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Initialize the adam optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        """Select an action based on the epsilon-greedy policy."""
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            # Randomly choose an action from the action space [0-3]
            return np.random.randint(0, self.action_dim)
        else:
            # Get the state as a tensor
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Predict the Q-values for the state using the Q-network
                action_values = self.q_network(state)
            # Choose the action with the highest Q-value by returning the index of the action
            # with the highest Q-value
            return np.argmax(action_values.cpu().data.numpy())

    def train(self):
        """Train the agent using the DQN algorithm. The agent samples a batch of experiences
        from the replay buffer, computes the Q-values for the current state and next state,
        and updates the Q-network weights. The target network is updated and the epsilon value
        is decayed."""
        # If the replay buffer is not full enough, do not train
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        experience = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experience)

        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Move the lists to a PyTorch tensor on the CPU/GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Predict the Q-values for the current state using the Q-network
        q_values = self.q_network(states)
        # Get the Q-values for the actions taken
        q_values = q_values.gather(1, actions)

        # Predict the Q-values for the next state using the target network
        next_q_values = self.target_network(next_states)
        # Get the maximum Q-value for the next state and unsqueeze it
        next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        # Compute the target q values using the Bellman equation. If the episode is done,
        # the future reward is 0 which is accounted for by (1 - dones)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate the loss using the mean squared error loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Clear the gradients
        self.optimizer.zero_grad()
        # Compute the gradients
        loss.backward()
        # Update the q-network weights
        self.optimizer.step()

        # Decay the epsilon value if the epsilon is greater than the epsilon min
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Increment the step count
        self.step_count += 1
        # Update the target network every 10 steps
        if self.step_count % self.target_update == 0:
            self.update_target_network()

    # Update the target network by copying the weights from the Q-network
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Save the model
    def save_model(self, file_path):
        torch.save(self.q_network.state_dict(), file_path)

    # Load the model
    def load_model(self, file_path):
        self.q_network.load_state_dict(torch.load(file_path))
        self.update_target_network()
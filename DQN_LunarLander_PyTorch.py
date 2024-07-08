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
    def __init__(self, state_dim, action_dim, replay_buffer, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_update = 10
        self.step_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    # Select an action based on the epsilon-greedy policy
    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.q_network(state)
            return np.argmax(action_values.cpu().data.numpy())

    # Train the agent by sampling experiences from the replay buffer
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        experience = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experience)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Predict the Q-values for the current state
        q_values = self.q_network(states).gather(1, actions)

        # Predict the Q-values for the next state
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)

        # Update the Q-values using the Bellman equation
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()

    # Update the target network
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Save the model
    def save_model(self, file_path):
        torch.save(self.q_network.state_dict(), file_path)

    # Load the model
    def load_model(self, file_path):
        self.q_network.load_state_dict(torch.load(file_path))
        self.update_target_network()
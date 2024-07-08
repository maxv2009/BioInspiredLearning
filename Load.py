import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# Define the Q-network again
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load the environment
env = gym.make('LunarLander-v2', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the network and load the trained parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qnetwork = QNetwork(state_size, action_size).to(device)
qnetwork.load_state_dict(torch.load('dqn_lunar_lander.pth'))
qnetwork.eval()
print("Model loaded successfully!")

# Function to select action based on the trained model
def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action_values = qnetwork(state)
    return np.argmax(action_values.cpu().data.numpy())

# Run the environment using the trained model
state = env.reset()[0]
total_reward = 0

while True:
    env.render()  # Render the environment to visualize
    action = select_action(state)
    next_state, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    state = next_state
    if done or truncated:
        break

print(f"Total reward: {total_reward}")
env.close()

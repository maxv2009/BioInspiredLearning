import matplotlib.pyplot as plt
import pandas as pd


def plot_rewards(episodes, rewards):
    # Calculate rolling average
    window_size = 10
    rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual rewards as bars
    ax.bar(episodes, rewards, color='lightblue', label='Individual Rewards')

    # Plot rolling average as a line
    ax.plot(episodes, rolling_avg, color='red', label='Rolling Average', linewidth=2)

    # Adding labels and title
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_title('Reward over Episodes with Rolling Average')
    ax.legend()

    # Display the plot
    plt.show()


def plot_save():
    plt.savefig('reward_plot.png')

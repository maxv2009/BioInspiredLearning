import matplotlib.pyplot as plt
import pandas as pd
import pickle


def load_lists(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_lists(filename, lists):
    with open(filename, 'wb') as f:
        pickle.dump(lists, f)


def plot_rewards(episodes, rewards, train_or_play, gamma="N/A", epsilon="N/A", epsilon_decay="N/A",
                 epsilon_min="N/A", batch_size="N/A", learning_rate="N/A", buffer_size="N/A",
                 save=False, foldername=None):
    # Calculate rolling average
    window_size = 200
    rolling_avg = pd.Series(rewards).rolling(window=window_size).mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Individual rewards as bars with no space between them
    ax.bar(episodes, rewards, color='lightblue', label='Individual Rewards', width=1.0)

    # Rolling average as a line
    ax.plot(episodes, rolling_avg, color='red', label='Rolling Average', linewidth=2)

    if train_or_play == "play":
        # Line at 200 points
        ax.axhline(y=200, color='black', linestyle='--', label='Solved Threshold')
    else:
        # Line at 250 points
        ax.axhline(y=240, color='black', linestyle='-', label='Rolling Average Threshold')
        # Line at 200 points
        ax.axhline(y=200, color='black', linestyle='--', label='Solved Threshold')

    # Adding labels and title
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_title('Reward over Episodes with Rolling Average')
    ax.legend()

    # Adding hyperparameters to the plot
    hyperparams_text = (
        f'Gamma: {gamma}, Epsilon: {epsilon}, Epsilon Decay: {epsilon_decay}, '
        f'Epsilon Min: {epsilon_min}, Batch Size: {batch_size}, '
        f'Learning Rate: {learning_rate}, Buffer Size: {buffer_size}'
    )
    plt.gcf().text(0.5, 0.02, hyperparams_text, fontsize=10, ha='center')

    if save:
        plt.savefig(f'.\\{foldername}\\reward_plot.eps', format='eps')

    plt.show()


def plot_multiple_rewards(all_rewards, all_episodes, all_names, foldername,
                          learning_rate, save=True):
    # Calculate rolling average for each set of rewards
    window_size = 200
    rolling_avg = [pd.Series(rewards).rolling(window=window_size).mean() for rewards in all_rewards]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(all_rewards)):
        # Rolling average as a line
        ax.plot(all_episodes[i], rolling_avg[i], label=all_names[i], linewidth=2)

    # Adding labels and title
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_title('Rolling Average for Different Hyperparameters')
    #ax.legend()

    if save:
        plt.savefig(f'.\\{foldername}\\reward_plot_lr_{learning_rate}.eps', format='eps')

    plt.show()  # Show the plot


# Plotting from saved data

#all_rewards, all_episodes, all_names = load_lists(f"Plot all "
#                                                 "hyperparameters\\repated_run_all_lists_lr_0.001.pkl")

#plot_multiple_rewards(all_rewards, all_episodes, all_names, "Plot all hyperparameters", 0.00001)
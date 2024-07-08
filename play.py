import gymnasium as gym
from gymnasium.utils import play

def main():
    env = gym.make('LunarLander-v2', render_mode="human")
    gym.utils.play.play(env, zoom=3)

if __name__ == "__main__":
    main()

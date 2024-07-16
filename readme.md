# Deep Q-Learning Applied to the Lunar Lander Environment
This code allows the user to train an agent to solve the lunar lander environment from the
Gymnasium package. The agent is trained using the DQN algorithm. The code is based on the paper
published by DeepMind in 2015.

Alternatively the user can load a pre-trained model and watch the agent solve the environment.

## Requirements
The required packages are listed in the requirements.txt file. To install the required packages 
run the following command: `pip install -r requirements.txt`

Depending on which package the user wants to use, either 'PyTorch' or 'TensorFlow' needs to be 
installed. It is recommended to use the 'PyTorch' package as it is the package used for the 
evaluation of the code and results in significant performance improvements.

## Usage
The user can either train an agent from scratch or load a pre-trained model.
### Playing
To load a pre-trained model, the user needs to set the train_or_play variable to "play" in the 
Train_DQN.py file. The user can then set the package to be used for the neural networks by 
setting the chosen_framework variable to either "PyTorch" or "TensorFlow". The user then needs 
to provide the path to the pre-trained model in the model_path variable.

The user can then run the model by executing the Train_DQN.py file.

The code will then load the pre-trained model and run the agent in the environment for the 
number of episodes specified in the num_episodes variable. The environment is loaded in the 
render mode, so the user can see the agent play the lunar lander. 
Every 100 episodes, the user will see a plot of the rewards obtained by the agent in each 
episode and the rolling average over the last 200 episodes.

### Training
To train an agent from scratch, the user needs to set the train_or_play variable to "train" in 
the Train_DQN.py file. The user can then set the package to be used for the neural networks by 
setting the chosen_framework variable to either "PyTorch" or "TensorFlow". The user then needs 
to provide the hyperparameters by setting the hyperparameter variables. 

The agent will then be trained for the number of episodes specified in the num_episodes 
variable. The environment is not rendered during the training to improve performance.

The model is saved every 100 steps during training and every 25 steps a plot of the rewards 
obtained by the agent in each episode and the rolling average over the last 200 episodes is 
shown.

Training concludes when the rolling average over the last 200 episodes is greater than 240 points.
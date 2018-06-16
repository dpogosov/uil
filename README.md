# Unsupervised imitation learning
Implementation of the final projecto of a university course

Unsupervised imitation learning - an apporach to exploration in deep reinforcement learning systems. 
The intuition here is that under uncertainty conditions an RL agent samples a possible action from the behavioral statistics of other  objects with lower uncertainty, pro rata to behavioral resemblance measure. We emphasize that all the objects may act as experts. An object with stronger statistical correlation between the given input and the coupled behaviour as well as with higher resemblance is more likely to "act" as an expert.

Refer to the *uil_pogosov.pdf* for further details.

## Requirements
* tensorflow 1.+
* keras 2.+
* python 3.5.0 or anaconda3 4.2.2
* openai gym (the latest)

## Files description

### Experiment control files
run_experiment_toy_v1.py - the main file that controls the experiment (requires pretrained LSTM, the weight files are included)
run_train_lstm_v1.py - trains LSTM
run_ddpg.py - runs DDPG under the toy environment

### Components and tools
controller.py - motion controller (controls conductor, i.e. provides ground truth for LSTM, performs the agent locomotion (action))
predictor.py - interface to LSTM predictor (requires pretrained LSTM)
lstm.py - LSTM predictor itself
toy_environment_v1.py - the toy environment (OpenAI interface + extras)
svg.py - the code of SVG
ddpg - the code of DDPG
explorer.py - Ornstein-Uhlenbeck process
net.py - neural networks for SVG and DDPG
replay_buffer.py - replay memory for RL algorithms

### Folders
save_lstm - folder for LSTM weights (there are the last weights)
save_svg - folder for SVG weights

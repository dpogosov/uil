import tensorflow as tf
from toy_environment_v1 import ToyEnvironmentV1
from net import StochasticNet
from svg import SVG
from replay_buffer import SimpleReplayPool
from predictor import Predictor

max_steps_per_episode = 350

# obs_include_conductor - includes the conductor position to the agent observation: False for SVG+LSTM, True otherwise
env = ToyEnvironmentV1(stp_max=max_steps_per_episode, target=[0., -0.5], obs_include_conductor=False, agent_random_start=False)
action_space = env.action_space.shape[0]
observation_space = env.observation_space.shape[0]

# pred_length = 13+ was ok. Too far, because of 'to point' controller (U is proportional to error)
predictor = Predictor(obs_length=10, pred_length=14)

# h_dim 100 was ok, batch normalisation does not improve
net = StochasticNet(action_space, observation_space, h_dim=100, on_bn=0)

# 1000 samples was ok
replay_buffer = SimpleReplayPool(1000, observation_space, action_space)

svg = SVG(epoch=500, min_buffer_size=256, pi_lr=1e-3, q_lr=1e-4, scale_reward=2.5, batch_size=64, tau=0.01,
          sess=tf.Session(), env=env,  max_path_length=max_steps_per_episode, predictor=predictor, net=net,
          action_space=action_space, observation_space=observation_space, replay_buffer=replay_buffer)

# save_every - does not work - keep higher than the number of epoch
# steps_to_solve - defines for how many steps in succession without sampling from LSTM a run is considered solved
# resample - switch on sampling from the predictor
svg.train(plot_every=5, save_every=10000, resample=True, steps_to_solve=10)
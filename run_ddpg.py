import gym
from gym import wrappers
import tensorflow as tf
import os
import argparse
#from rllab.envs.gym_env import GymEnv

from net import DeterministicNet
from ddpg import DDPG
from explorer import OUStrategy
from replay_buffer import SimpleReplayPool
from toy_environment_v1 import ToyEnvironmentV1

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='log_ddpg')
args = parser.parse_args()

env = ToyEnvironmentV1(target=[0., -0.5])
env_spec = dict(
        action_space=env.action_space,
        observation_space=env.observation_space)

net = DeterministicNet(env_spec, h_dim=100)

replay_buffer = SimpleReplayPool(1000000, env_spec['observation_space'].shape[0], env_spec['action_space'].shape[0])

ou = OUStrategy(env_spec)

config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
sess = tf.Session(config=config)

agent = DDPG(epoch=100000, net=net, env_spec=env_spec, replay_buffer=replay_buffer, es=ou, sess=sess, env=env, scale_reward=1., max_path_length=350, batch_size=32, num_critic_update=10)
agent.train_step()

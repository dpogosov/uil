import tensorflow as tf
import os
import pickle
from lstm import LSTM


class Predictor(object):
    def __init__(self, obs_length=5, pred_length=3):
        self.obs_length = obs_length
        self.pred_length = pred_length

        # Load the saved arguments to the model from the config file
        with open(os.path.join('save_lstm', 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)

        # Initialize with the saved args
        self.model = LSTM(saved_args, True)
        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()

        # Get the checkpoint state to load the model from
        ckpt = tf.train.get_checkpoint_state('save_lstm')
        print('loading model: ', ckpt.model_checkpoint_path)

        # Restore the model at the checpoint
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, path, full=False):
        obs_traj = path[:self.obs_length]  # observed part of the trajectory
        # Get the complete trajectory with both the observed and the predicted part from the model
        predicted_traj, mu, var = self.model.sample(self.sess, obs_traj, num=self.pred_length, full=full)
        if full:
            return predicted_traj, mu, var
        else:
            return predicted_traj

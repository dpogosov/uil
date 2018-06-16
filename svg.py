import sys
import tensorflow as tf
import numpy as np
from keras.layers.merge import Concatenate
from keras.layers import Input
import keras.backend as K
from keras.models import load_model
import timeit
from predictor import Predictor
from controller import Controller


class SVG(object):
    def __init__(self, epoch, net, action_space, observation_space, replay_buffer, batch_size=32, min_buffer_size=10000,
                 max_path_length=200, pi_lr=1e-3, q_lr=1e-4, sess=None, tau=0.001, gamma=0.99, scale_reward=1.0,
                 env=None, predictor=None):
        self.predictor = predictor
        self.epoch = epoch
        self.net = net
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.max_path_length = max_path_length
        self.in_dim = observation_space
        self.action_dim = action_space
        self.pi_opimizer = tf.train.AdamOptimizer(pi_lr)
        self.q_optimizer = tf.train.AdamOptimizer(q_lr)
        self.sess = sess
        self.tau = tau
        self.gamma = gamma
        self.scale_reward = scale_reward
        self.env = env
        self.built = False
        self.path_predicted = []

    def build(self):
        model = self.net.model
        mu_model = self.net.mu_model
        var_a_model = self.net.var_a_model
        q_model = self.net.q_model
        target_model = self.net.target_model
        target_mu_model = self.net.target_mu_model
        target_var_a_model = self.net.target_var_a_model
        target_q_model = self.net.target_q_model

        self.states = tf.placeholder(tf.float32, shape=(None, self.in_dim), name='states')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.action_dim], name='actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')
        self.next_states = tf.placeholder(tf.float32, shape=[None, self.in_dim], name='next_states')
        self.ys = tf.placeholder(tf.float32, shape=[None])
        # There are other implementations about how can we take aciton.
        # Taking next action version or using only mu version or searching action which maximize Q.
        target_mu = target_mu_model(self.states)
        target_var_a = target_var_a_model(self.states)
        target_action = target_mu + K.random_normal(K.shape(target_mu), dtype=tf.float32) * K.exp(target_var_a)
        self.target_q = K.sum(target_q_model(Concatenate()([target_model(self.states), target_action])), axis=-1)
        self.q = K.sum(q_model(Concatenate()([model(self.states), self.actions])), axis=-1)
        self.q_loss = K.mean(K.square(self.ys-self.q))
        self.mu = mu_model(self.states)
        self.var_a = var_a_model(self.states)
        self.eta = (self.actions - self.mu) / K.exp(self.var_a)
        inferred_action = self.mu + K.stop_gradient(self.eta) * K.exp(self.var_a)
        self.pi_loss = - K.mean(q_model(Concatenate()([model(self.states), inferred_action])))
        self.q_updater = self.q_optimizer.minimize(self.q_loss, var_list=self.net.var_q)
        self.pi_updater = self.pi_opimizer.minimize(self.pi_loss, var_list=self.net.var_pi)
        self.soft_updater = [K.update(t_p, t_p*(1-self.tau)+p*self.tau) for p, t_p in zip(self.net.var_all, self.net.var_target_all)]
        self.sync = [K.update(t_p, p) for p, t_p in zip(self.net.var_all, self.net.var_target_all)]
        self.sess.run(tf.global_variables_initializer())
        self.built = True

    def optimize_q(self, batch):
        if not self.built:
            self.build()
        next_q = self.sess.run(self.target_q, {self.states: batch['next_states'], K.learning_phase(): 1})
        ys = batch['rewards'] + (1 - batch['terminals']) * next_q * self.gamma
        feed_in = {
                self.states: batch['states'],
                self.actions: batch['actions'],
                self.rewards: batch['rewards'],
                self.next_states: batch['next_states'],
                self.ys: ys,
                K.learning_phase(): 1
                }
        self.sess.run(self.q_updater, feed_in)

    def optimize_pi(self, batch):
        if not self.built:
            self.build()
        feed_in = {
                self.states: batch['states'],
                self.actions: batch['actions'],
                K.learning_phase(): 1
                }
        self.sess.run(self.pi_updater, feed_in)

    def get_action(self, state):
        mu = self.sess.run(self.mu, {self.states: [state], K.learning_phase(): 0})[0]
        var_a = self.sess.run(self.var_a, {self.states: [state], K.learning_phase(): 0})[0]
        a = mu + np.random.normal(size=mu.shape) * np.exp(var_a)
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        return a, var_a

    def test(self, plot_every=1):
        if not self.built:
            self.build()
        self.load_weights(550)
        self.sess.run(self.sync)
        for i in range(0, 10):
            s = self.env.reset()
            total_reward = 0
            terminal = False
            max_path_length = self.max_path_length
            for step in range(max_path_length):
                if terminal:
                    break
                a, agent_info = self.get_action(s)
                next_s, r, terminal, env_info = self.env.step(a)
                total_reward += r
                s = next_s
                if not i % plot_every:
                    self.env.render()
            print('total reward:', total_reward, ' episode:', i)

    def train(self, plot_every=10, save_every=100, resample=False, true_controller=False, steps_to_solve=10):

        tru = Controller(vj=0.75, dt=0.05, u_max=2., radius=1., A=0.1)  # DEBUG only

        reached = 0
        solved = 0
        solved_max = 0

        if not self.built:
            self.build()
        self.sess.run(self.sync)
        min_buf_fill = False

        for i in range(self.epoch):
            dP = np.array([0.001, 0.001])

            s = self.env.reset()
            total_reward = 0
            terminal = False

            step_gamma = 1
            a_p = 0
            resample_counter = 0

            if solved > steps_to_solve:
                print('SOLVED')
                break

            for step in range(self.max_path_length):
                do_resample = 0

                if terminal:
                    break
                a, var_a = self.get_action(s)

                # normalization variance to probability
                clipped_var = np.clip(var_a, -1, 0)
                prob_action = (np.exp(clipped_var)-0.367)/0.633

                var_p = np.zeros((self.predictor.obs_length+self.predictor.pred_length, 2))

                for index, var in enumerate(prob_action):
                    if (np.random.random() < var) and (step > self.predictor.obs_length) and resample:
                        do_resample = 1
                self.env.reset_magic()

                if do_resample:
                    resample_counter += 1
                    step_gamma = 0  # mixing iterative learning reset condition
                    if true_controller:  # DEBUG only
                        err, a_p, dP = tru.point(np.asarray(s[0:2]), dP, self.env.target, magic=75.)
                    else:  # NORMAL operation
                        self.env.magic = 4.
                        path = self.env.get_path(self.predictor.obs_length)
                        self.path_predicted, mu_p, var_p = self.predictor.predict(path, full=True)
                        a_p = self.path_predicted[self.predictor.obs_length+self.predictor.pred_length-1]

                # mixing iterative learning
                gamma_decayed = self.gamma ** step_gamma
                a = a * (1. - gamma_decayed) + a_p * gamma_decayed
                step_gamma += 1

                next_s, r, terminal, env_info = self.env.step(a)
                total_reward += r

                self.replay_buffer.add_sample(s, a, r*self.scale_reward, terminal)

                if self.replay_buffer.size >= self.min_buffer_size:
                    if not min_buf_fill:
                        print('minimum buffer filled')
                        min_buf_fill = True
                    batch = self.replay_buffer.random_batch(self.batch_size)
                    self.optimize_q(batch)
                    self.optimize_pi(batch)
                self.sess.run(self.soft_updater)

                # plotting
                if (not i % plot_every):
                    self.env.render()
                    if step > self.predictor.obs_length:
                        self.env.render_predicted(self.path_predicted)

            # whether the run is solved (conditions)
            if total_reward and not resample_counter:
                solved += 1
            else:
                solved = 0
            if solved > solved_max:
                solved_max = solved
            if total_reward:
                reached += 1

            print('episode:', i, '  reward:', total_reward, '  resampled:', resample_counter, 'of', step,
                  '  solved: %.0f%%' % (100 * solved/steps_to_solve),
                  '(max: %.0f%%)' % (100 * solved_max/steps_to_solve), flush=True)

            # save weights
            if (not i % save_every) and (i > 0):
                print('save_old weights at', i)
                self.net.model.save('save_svg/model_' + str(i) + '.h5')
                self.net.q_model.save('save_svg/q_model_' + str(i) + '.h5')
                self.net.pi_model.save('save_svg/pi_model_' + str(i) + '.h5')
                self.net.mu_model.save('save_svg/mu_model_' + str(i) + '.h5')
                self.net.var_a_model.save('save_svg/var_a_model_' + str(i) + '.h5')

                self.net.target_model.save('save_svg/target_model_' + str(i) + '.h5')
                self.net.q_model.save('save_svg/target_q_model_' + str(i) + '.h5')
                self.net.target_pi_model.save('save_svg/target_pi_model_' + str(i) + '.h5')
                self.net.target_mu_model.save('save_svg/target_mu_model_' + str(i) + '.h5')
                self.net.target_var_a_model.save('save_svg/target_var_a_model_' + str(i) + '.h5')
        print('Reached', reached, '/', self.epoch)

    def load_weights(self, episode, targets=False):
        i = episode
        self.net.model = load_model('save_svg/model_'+str(i)+'.h5')
        self.net.q_model = load_model('save_svg/q_model_' + str(i) + '.h5')
        self.net.pi_model = load_model('save_svg/pi_model_'+str(i)+'.h5')
        self.net.mu_model = load_model('save_svg/mu_model_'+str(i)+'.h5')
        self.net.var_a_model = load_model('save_svg/var_a_model_'+str(i)+'.h5')
        if targets:
            self.net.target_model = load_model('save_svg/target_model_'+str(i)+'.h5')
            self.net.q_model = load_model('save_svg/target_q_model_'+str(i)+'.h5')
            self.net.target_pi_model = load_model('save_svg/target_pi_model_'+str(i)+'.h5')
            self.net.target_mu_model = load_model('save_svg/target_mu_model_'+str(i)+'.h5')
            self.net.target_var_a_model = load_model('save_svg/target_var_a_model_'+str(i)+'.h5')

    def tic(self):
        self._tic = timeit.default_timer()

    def toc(self):
        toc = timeit.default_timer()
        print('Time elapsed', toc - self._tic)
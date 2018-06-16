# model
# t*dx = U - k*x

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from controller import Controller

class ToyEnvironmentV1(object):
    def __init__(self, dt=0.01, a_max=2.5, stp_max=30000, target=[0., 0.], radius=0.125, obs_include_conductor=True, agent_random_start=False):
        if obs_include_conductor:
            self.observation_space = spaces.Box(low=-3, high=3, shape=6)
        else:
            self.observation_space = spaces.Box(low=-3, high=3, shape=4)
        self._obs_include_conductor = obs_include_conductor

        self.action_space = spaces.Box(low=-a_max, high=a_max, shape=2)

        self.agent_random_start=agent_random_start
        self.dt = dt
        self.a_max = a_max
        self.stp_max = stp_max
        self.target = np.asarray(target)
        self.control = Controller(u_max=1., dt=self.dt, radius=1.22)
        self.conductor = Controller(dt=dt, u_max=a_max, radius=1.22, seq_length=5, render=False, num_batches=10,
                                    batch_size=2, pred_length=3, to_point=True, target=target, no_random=False,
                                    reached=radius)
        self.radius = radius

        self.reached = 0
        self.episode = 0
        self.magic = 1.5

        plt.ion()
        self.reset()


    def step(self, action):
        info = None
        done = False
        # get new action
        action_cipped = np.clip(action, -self.a_max, self.a_max)
        _, self.x, self.dx = self.control.point(self.x, self.dx, action_cipped, magic=self.magic)
        # inaccuracy in action is punishment
        toofar = self.control.distance(action, self.x)
        distance = self.control.distance(self.x, self.target)
        # TODO switch on penalizing for too far action
        reward = -toofar*0

        self.path += distance*self.dt
        observation = np.append(self.x, self.dx)

        # adding behaviour of the conductor object
        err_cond, self.x_cond, self.dx_cond = self.control.point(self.x_cond, self.dx_cond, self.target, magic=2.05)
        if err_cond < 0.025:  # kill vagrancy
            self.x_cond = self.target
            self.dx_cond = [0, 0]
        if self._obs_include_conductor:
            observation = np.append(observation, self.x_cond)

        # end episode conditions
        error = self.control.distance(self.x, self.target)
        if error < self.radius:
            done = True
            reward += 100 - 2.*(self.path + self.control.distance(self.started, self.target)*10.)
            self.reached += 1
        if abs(self.x[0]) > 2.5 or abs(self.x[1]) > 2.5:
            done = True
            reward -= 100
        self.stp += 1
        if self.stp > self.stp_max:
            done = True
        self.accumulated_r += reward

        # save history
        self.x_hist.append(self.x[0])
        self.y_hist.append(self.x[1])
        self.path_hist = np.append(self.path_hist, self.x, axis=0)
        self.x_cond_hist.append(self.x_cond[0])
        self.y_cond_hist.append(self.x_cond[1])
        self.path_cond_hist = np.append(self.path_cond_hist, self.x_cond, axis=0)

        return observation, reward, done, info

    def reset(self):
        self.x_hist = []
        self.y_hist = []
        self.x_cond_hist = []
        self.y_cond_hist = []
        self.path_hist = []
        self.path_cond_hist = []
        self.path=0.
        self.reached = 0
        if self.agent_random_start:
            self.x = self.control.set_random_position()
        else:
            self.x = np.array([1, -0.7])
        self.started = self.x
        self.dx = [0, 0]
        self.dx_cond = [0, 0]
        self.accumulated_r = 0
        self.stp = 0
        self.x_cond = self.conductor.set_random_position()
        self.episode += 1
        observation = np.append(self.x, self.dx)
        if self._obs_include_conductor:
            observation = np.append(observation, self.x_cond)
        plt.cla()
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.grid(True)
        ax = plt.gca()
        circle = plt.Circle( (self.target[0], self.target[1]) , self.radius, color='k', fill=False)
        ax.add_artist(circle)
        circle = plt.Circle((0, 0), 1.22, color='k', fill=False)
        ax.add_artist(circle)
        return observation

    def render(self, render_conductor=True):
        # animation
        if (not self.stp % 5):
            plt.plot(self.x_hist, self.y_hist, 'r-', linewidth=1.5)  # , markersize=4)
            if render_conductor:
                plt.plot(self.x_cond_hist, self.y_cond_hist, 'g-', linewidth=1.5)
            d = (10.*self.control.distance(self.x, self.target))**2
            plt.title("reached %i/%i, reward %.2f, distance %.2f, timeout %.2f" % (self.reached, self.episode, self.accumulated_r, d, self.stp/self.stp_max))
            red_patch = mpatches.Patch(color='red', label='Actual')
            green_patch = mpatches.Patch(color='green', label='Conductor')
            blue_patch = mpatches.Patch(color='blue', label='Prediction')
            plt.legend(handles=[red_patch, green_patch, blue_patch])
            #plt.legend(handles=[green_patch])
            plt.pause(1e-10)

    def get_path(self, length):
        pl = len(self.path_hist)//2
        P_hist = np.reshape(self.path_hist, (pl, 2))
        return P_hist[pl-length:pl, :]

    def get_path_cond(self, length):
        pl = len(self.path_cond_hist)//2
        P_hist = np.reshape(self.path_cond_hist, (pl, 2))
        return P_hist[pl-length:pl, :]

    def render_predicted(self, path):
        path = np.asanyarray(path)
        if (not self.stp % 5) and path.any():
            plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=1.0)

    def reset_magic(self):
        self.magic = 1.5
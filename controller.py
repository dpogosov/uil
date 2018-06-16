# simple controller: just cyrcle, and point, no orientation
# model ddp = u

import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl


class Controller(object):
    def __init__(self, vj=0.75, dt=0.05, u_max=1., radius=1., A=0.1, seq_length=5, render=False, xy_shift=1,
                 num_batches=10, batch_size=2, pred_length=3, to_point=True, target=[0., 0.],
                 no_random=False, reached=0.13):
        self.seq_length = seq_length
        self.vj = vj
        self.r = radius
        self.dt = dt
        self.u_max = u_max
        self.A = A  # tuning
        self.render = render
        self.xy_shift = xy_shift
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.pred_length = pred_length
        self.u = []
        self._to_point = to_point
        self.target = np.array(target)
        self._no_random = no_random
        self.P_init = [0., 0.]
        self.reached = reached
        plt.ion()
        self.reset_batch_pointer()

    def reset_batch_pointer(self):
        if self.render:
            plt.cla()
            plt.xlim(-2. * self.r, 2. * self.r)
            plt.ylim(-2. * self.r, 2. * self.r)
            plt.grid(True)
            ax = plt.gca()
            circle = plt.Circle((0, 0), self.r, color='k', fill=False)
            ax.add_artist(circle)
        self.P = np.array([self.r, 0.])
        self.dP = np.array([0.001, 0.001])

    def next_batch(self):
        x_batch = []
        y_batch = []
        terminal = True
        overstay, overstayp = 0, 0
        for b in range(0, self.batch_size):
            if terminal:
                terminal = False
                if self._no_random:
                    self.P = np.array(self.P_init)
                else:
                    self.set_random_position()
                self.dP = np.array([0.001, 0.001])
                overstayp = overstay
                overstay = 0
            P_hist = []
            x_hist = []
            y_hist = []
            for t in range(0, self.seq_length + self.xy_shift):
                if self._to_point:
                    psi, self.P, self.dP = self.point(self.P, self.dP, self.target)
                else:
                    psi, self.P, self.dP = self.control(self.P, self.dP)
                    psi = psi[0]
                if psi < self.reached:
                    terminal = True
                    overstay +=1
                x_hist.append(self.P[0])
                y_hist.append(self.P[1])
                P_hist = np.append(P_hist, self.P, axis=0)
                if self.render:
                    plt.plot(x_hist, y_hist, 'r-', linewidth=1.5)
                    plt.title("error %.3f, overstay %i" % (psi, overstayp))
                    plt.pause(1e-10)
            P_hist = np.reshape(P_hist, (self.seq_length + self.xy_shift, 2))
            x_batch.append(np.copy(P_hist[0:self.seq_length, :]))
            y_batch.append(np.copy(P_hist[self.xy_shift:self.seq_length + self.xy_shift, :]))
        return x_batch, y_batch

    def render_predicted(self, path, forced=False):
        if self.render or forced:
            plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=1.0)
            plt.pause(1e-10)

    def control(self, P, dP):  # trajectory control (circle)
        psi = np.array([np.matmul(np.transpose(P), P) - self.r ** 2, 0.])
        psiv = np.array([0., np.matmul(dP, np.transpose(dP)) - self.vj ** 2])
        J = (self.A ** 2.) * np.array([2. * P, [0., 0.]])
        Jv = np.array([[0., 0.], 2 * dP])
        G = (self.A ** 2.) * np.array([2. * dP, [0., 0.]])
        u = - np.matmul(npl.inv(J + Jv), (2. * np.matmul(2. * J / self.A + G, dP) + psi + psiv))
        u = np.clip(u, -self.u_max, self.u_max)
        dP += u * self.dt
        P += dP * self.dt
        return psi, P, dP

    def point(self, P, dP, target, magic=1.):  # to point control
        self.u = target - P - 2. * np.asarray(dP)
        self.u = np.clip(self.u, -self.u_max, self.u_max)
        dP += self.u * self.dt
        P += dP * self.dt * magic
        err = self.distance(P, target)
        return err, P, dP

    def distance(self, A, B):
        d = np.sqrt(np.sum(np.power(np.asarray(A) - np.asarray(B), 2)))
        return d

    def set_random_position(self):  # random put a point on a circle
        self.P[0] = (2. * rnd.random() - 1.) * self.r
        self.P[1] = np.sqrt(self.r ** 2 - self.P[0] ** 2) * np.sign(rnd.random() - 0.5)
        return self.P






import cppimport
import numpy as np
from tqdm import tqdm
import scipy as sc
import scipy.integrate as integrate
import pandas as pd

cpp_hm = cppimport.imp("c_henon_map")

class henon_radial(object):
    def __init__(self, n_theta, n_steps, epsilon):
        self.n_theta = n_theta
        self.n_steps = n_steps
        self.epsilon = epsilon

        self.engine = cpp_hm.henon_radial(n_theta, n_steps, epsilon)
        self.times = self.engine.compute(0, 1)
        self.times = np.array(self.times)

    def reset(self):
        self.engine.reset()
        self.times = self.engine.compute(0, 1)
        self.times = np.array(self.times)

    def compute(self, n_iterations):
        self.times = self.engine.compute(n_iterations, 1)
        self.times = np.array(self.times)
        return pd.DataFrame(self.times.reshape((self.n_theta, self.n_steps)), index=np.linspace(0, np.pi/2, self.n_theta), columns=np.linspace(0, 1.0, self.n_steps))

    def get_times(self):
        return pd.DataFrame(self.times.reshape((self.n_theta, self.n_steps)), index=np.linspace(0, np.pi/2, self.n_theta), columns=np.linspace(0, 1.0, self.n_steps))

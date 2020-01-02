import cppimport
import numpy as np
from tqdm import tqdm
import scipy as sc
import scipy.integrate as integrate
import pandas as pd

import c_henon_map as cpp_hm

class henon_radial(object):
    def __init__(self, n_theta, n_steps, epsilon):
        """Init a henon_radial analysis tool
        
        Parameters
        ----------
        object : self
            self
        n_theta : int
            number of angles to scan between 0 and pi / 2
        n_steps : int
            number of positions to scan between 0 and 1
        epsilon : float
            intensity of the modulation (see references)
        """        
        self.n_theta = n_theta
        self.n_steps = n_steps
        self.epsilon = epsilon

        self.engine = cpp_hm.henon_radial(n_theta, n_steps, epsilon)
        self.times = self.engine.compute(0, 1)
        self.times = np.array(self.times)

    def reset(self):
        """Reset the engine to the initial conditions
        """        
        self.engine.reset()
        self.times = self.engine.compute(0, 1)
        self.times = np.array(self.times)

    def compute(self, n_iterations):
        """Compute n iterations of the map
        
        Parameters
        ----------
        n_iterations : int
            number of iterations
        
        Returns
        -------
        Dataframe
            Dataframe of the angle-distance times before particle loss.
        """        
        self.times = self.engine.compute(n_iterations, 1)
        self.times = np.array(self.times)
        return pd.DataFrame(self.times.reshape((self.n_theta, self.n_steps)), index=np.linspace(0, np.pi/2, self.n_theta), columns=np.linspace(0, 1.0, self.n_steps))

    def get_times(self):
        """Get the survival times of the molecules in a dataframe
        
        Returns
        -------
        Dataframe
            Dataframe of the angle-distance times before particle loss.
        """        
        return pd.DataFrame(self.times.reshape((self.n_theta, self.n_steps)), index=np.linspace(0, np.pi/2, self.n_theta), columns=np.linspace(0, 1.0, self.n_steps))

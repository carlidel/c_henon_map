import numpy as np
from tqdm import tqdm
import scipy as sc
import scipy.integrate as integrate
import pandas as pd

from . import c_henon_map as cpp_hm

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
        self.times = np.array(self.engine.compute(0, 1))

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


class henon_grid(object):
    def __init__(self, n_x, n_y, epsilon):
        """Init a henon_grid analysis tool

        Parameters
        ----------
        object : self
            self
        n_x : int
            number of x positions to scan
        n_y : int
            number of y positions to scan
        epsilon : float
            intensity of the modulation (see references)
        """
        self.n_x = n_x
        self.n_y = n_y
        self.epsilon = epsilon

        self.engine = cpp_hm.henon_grid(n_x, n_y, epsilon)
        self.times = np.array(self.engine.compute(0, 1))

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
        ndarray
            2d matrix of the times.
        """
        self.times = self.engine.compute(n_iterations, 1)
        self.times = np.array(self.times)
        return self.times.reshape((self.n_x, self.n_y))

    def get_times(self):
        """Get the survival times of the molecules in a matrix
        
        Returns
        -------
        ndarray
            2d matrix of the times.
        """
        return self.times.reshape((self.n_x, self.n_y))


class henon_scan(object):
    def __init__(self, x0, y0, px0, py0, epsilon):
        """Initialize custom henon-scan
        
        Parameters
        ----------
        object : self
            self
        x0 : ndarray
            x starting positions
        y0 : ndarray
            y starting positions
        px0 : ndarray
            px starting conditions
        py0 : ndarray
            py starting positions
        epsilon : float
            intensity of modulation (see references)
        """        
        self.x0 = x0
        self.y0 = y0
        self.px0 = px0
        self.py0 = py0
        self.epsilon = epsilon

        self.engine = cpp_hm.henon_scan(x0, y0, px0, py0, epsilon)
        self.times = np.array(self.engine.compute(0, 1)[2])

    def reset(self):
        """Reset the engine to the initial condition
        """        
        self.engine.reset()
        self.times = np.array(self.engine.compute(0, 1)[2])

    def compute(self, n_iterations):
        """Compute n iterations of the map
        
        Parameters
        ----------
        n_iterations : int
            number of iterations
        
        Returns
        -------
        (ndarray, ndarray, ndarray)
            x0, y0, times
        """
        data = self.engine.compute(n_iterations, 1)
        self.times = np.array(data[2])
        return np.array(data[0]), np.array(data[1]), np.array(data[2])

    def get_times(self):
        """Get the survival times of the molecules in a tuple
        
        Returns
        -------
        (ndarray, ndarray, ndarray)
            x0, y0, times
        """
        return self.x0, self.y0, self.times


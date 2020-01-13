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

    def get_data(self):
        """Get general data with also last stable orbit position.
        
        Returns
        -------
        tuple of 5 ndarrays
            x, y, px, py, times, lost boolean
        """
        data = self.engine.get_data()
        for coso in data:
            coso = coso.reshape((self.n_theta, self.n_steps))
        return np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]), np.array(data[4]), np.array(data[5])


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

    def get_data(self):
        """Get general data with also last stable orbit position.
        
        Returns
        -------
        tuple of 5 ndarrays
            x, y, px, py, times, lost boolean
        """
        data = self.engine.get_data()
        for coso in data:
            coso = coso.reshape((self.n_x, self.n_y))
        return np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]), np.array(data[4]), np.array(data[5])


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

    def get_data(self):
        """Get general data with also last stable orbit position.
        
        Returns
        -------
        tuple of 5 ndarrays
            x, y, px, py, times, lost boolean
        """        
        data = self.engine.get_data()
        return np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]), np.array(data[4]), np.array(data[5])


class henon_track(object):
    def __init__(self, x0, y0, px0, py0, epsilon):
        """init an henon full tracking object
        
        Parameters
        ----------
        object : self
            self
        x0 : float
            x starting point
        y0 : float
            y starting point
        px0 : float
            px starting point
        py0 : float
            py starting point
        epsilon : float
            modulation intensity (see references)
        """        
        self.x0 = x0
        self.y0 = y0
        self.px0 = px0
        self.py0 = py0
        self.epsilon = epsilon

        self.engine = cpp_hm.henon_scan(x0, y0, px0, py0, epsilon)
        self.times = 0

    def reset(self):
        """Resets the engine.
        """        
        self.engine.reset()
        self.times = 0
        
    def compute(self, iterations):
        """compute the tracking
        
        Parameters
        ----------
        iterations : unsigned int
            number of iterations
        
        Returns
        -------
        tuple of ndarrays
            (x, y, px, py)
        """        
        self.data = self.engine.compute(iterations)
        self.times += iterations
        return np.asarray(self.data[0]), np.asarray(self.data[1]), np.asarray(self.data[2]), np.asarray(self.data[3])

    def get_data(self):
        """Get the data
        
        Returns
        -------
        tuple of ndarrays
            (x, y, px, py)
        """        
        self.data = self.engine.get_data()
        return np.asarray(self.data[0]), np.asarray(self.data[1]), np.asarray(self.data[2]), np.asarray(self.data[3])


def cartesian_to_polar_4d(x, y, px, py):
    """Convert a 4d cartesian point to a 4d polar variable point.
    
    Parameters
    ----------
    x : ndarray
        ipse dixit
    y : ndarray
        ipse dixit
    px : ndarray
        ipse dixit
    py : ndarray
        ipse dixit
    
    Returns
    -------
    tuple of ndarray
        (r, alpha, theta1, theta2)
    """    
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(px, 2) + np.power(py, 2))
    theta1 = np.arctan2(px, x)
    theta2 = np.arctan2(py, y)
    alpha = np.arctan2(np.sqrt(y * y + py * py), np.sqrt(x * x + px * px))
    return r, alpha, theta1, theta2

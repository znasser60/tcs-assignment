import numpy as np 
import scipy as sp
import scipy.stats as ss 
import matplotlib.pyplot as plt

class Neuron: 
    def __init__(self, file_path): 
        """
        Initialize a Neuron object.
        """
        self.file_path = file_path 

    def load_data(self): 
        self.data = np.loadtxt(self.file_path)
    
    def plot_time_distribution(self): 
        """
        Plot the time distribution of the neuron firing.
        """
        self.difference = np.diff(self.data)
        plt.hist(self.difference, bins=50, density=True)
        plt.title("Time Distribution of a Neuron Firing")
        plt.xlabel("Time (ms)")
        plt.ylabel("Density")
        plt.show()
    
    def calc_refractory_period(self): 
        """
        Calculate the refractory period of the neuron.
        """
        self.refractory_period = np.min(self.difference)
        return self.refractory_period
    
    def fit_exponential(self): 
        """
        Fits an exponential function to the time distribution.
        """
        def exponential(x, a, b):
            return a * np.exp(-b * x)
        
        self.hist, self.bins = np.histogram(self.difference, bins=50, density=True)
        self.bin_centers = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.exp_params, self.exp_cov = sp.optimize.curve_fit(exponential, self.bin_centers, self.hist)
        r_squared = 1 - (np.sum((self.hist - exponential(self.bin_centers, *self.exp_params))**2) / np.sum((self.hist - np.mean(self.hist))**2))
        
        x_axis = np.linspace(0, 100, 100)
        plt.plot(x_axis, exponential(x_axis, *self.exp_params), label='Exponential Fit', color='red')
        plt.hist(self.difference, bins=50, density=True, label='Data')
        plt.title("Exponential Fit to Time Distribution")
        plt.xlabel("Time (ms)")
        plt.ylabel("Density")
        print("Exponential Fit Parameters: ", self.exp_params, "R^2: ", r_squared)

        t = np.linspace(0, 100, 100)
        self.inter_spike_distribution = 0.1*np.exp(-0.1 * t)
        plt.plot(t, self.inter_spike_distribution, label='Analytical Distribution')
        plt.legend()
        plt.show()
    

if __name__ == "__main__":
    neuron = Neuron("data/Data_neuron.txt")
    neuron.load_data()
    neuron.plot_time_distribution()
    print("Refractory period: ", neuron.calc_refractory_period())
    neuron.fit_exponential()
    neuron.inter_spike_distribution()

import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt

class Neuron: 
    def __init__(self, file_path): 
        """
        Initialize a Neuron object.
        """
        self.file_path = file_path 

    def load_data(self): 
        self.data = np.loadtxt(self.file_path)
    
    def plot_time_distribution(self, save_path=None): 
        """
        Plot the time distribution of the neuron firing.
        """
        self.difference = np.diff(self.data)
        plt.figure()
        plt.hist(self.difference, bins=50, density=True, color='b', edgecolor='black', linewidth=0.7, alpha=0.7)
        plt.title("Time Distribution of Neuron Firing", fontsize=14)
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(visible=True, alpha=0.5)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path)
        else: 
            plt.show()
    
    def calc_refractory_period(self): 
        """
        Calculate the refractory period of the neuron.
        """
        self.refractory_period = np.min(self.difference)
        return self.refractory_period
    
    def fit_exponential(self, save_path=None): 
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
        plt.figure()
        plt.plot(x_axis, exponential(x_axis, *self.exp_params), label='Exponential Fit', color='red', linewidth=1.5)
        plt.hist(self.difference, bins=50, density=True, label='Data', color='b', edgecolor='black', alpha=0.7)
        plt.title("Exponential Fit to Time Distribution", fontsize=14)
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Density", fontsize=12)

        t = np.linspace(self.refractory_period, 100, 500)
        self.inter_spike_distribution = self.exp_params[1] * np.exp(-self.exp_params[1] * (t - self.refractory_period))
        plt.plot(t, self.inter_spike_distribution, label='Analytical Distribution', linewidth=2, linestyle='--')
        plt.legend()
        plt.grid(visible=True, alpha=0.5)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path)
        else: 
            plt.show()

        print(f"Exponential Fit Parameters: {self.exp_params}, R^2: {r_squared}")
    
    def generate_spike_data(self, num_spikes=1000, save_path=None):
        """
        Generates 1000 new values based on the fitted exponential distribution 
        from the data. 
        """
        self.gen_spike_data = np.zeros(num_spikes)
        self.gen_spike_data = np.cumsum(self.refractory_period + np.random.exponential(1/self.exp_params[1], num_spikes))
        plt.figure()
        plt.hist(np.diff(self.gen_spike_data), bins=50, density=True, alpha=0.5, label="Generated", edgecolor='black', linewidth=0.7)
        plt.hist(self.difference, bins=50, density=True, alpha=0.5, label="Original", edgecolor='black', linewidth=0.7)
        plt.title("Original vs Generated Spike Interval Distribution", fontsize=14)
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(visible=True, alpha=0.5)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path)
        else: 
            plt.show()
        # # Quantify the difference between the original and generated data
        # ks_statistic, p_value = sp.stats.ks_2samp(self.difference, np.diff(self.gen_spike_data))
        # print(f"KS Statistic: {ks_statistic}, p-value: {p_value}")


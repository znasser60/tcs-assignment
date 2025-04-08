import numpy as np 
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

if __name__ == "__main__":
    neuron = Neuron("data/Data_neuron.txt")
    neuron.load_data()
    neuron.plot_time_distribution()
    print("Refractory period: ", neuron.calc_refractory_period())

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
        plt.title("Time Distribution of Neuron Firing")
        plt.xlabel("Time (ms)")
        plt.ylabel("Density")
        plt.show()

if __name__ == "__main__":
    neuron = Neuron("data/Data_neuron.txt")
    neuron.load_data()
    neuron.plot_time_distribution()

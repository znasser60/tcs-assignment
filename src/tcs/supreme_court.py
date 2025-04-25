import numpy as np 
import matplotlib.pyplot as plt

class SupremeCourt: 
    def __init__(self, hi_file_path, Jij_file_path, SC_file_path): 
        """
        Initialize a SupremeCourt object.
        """
        self.hi_file_path = hi_file_path 
        self.Jij_file_path = Jij_file_path 
        self.SC_file_path = SC_file_path 
        self.data = None

    def load_data(self): 
        """
        Load data from the file path.
        """
        with open(self.SC_file_path, 'r') as f:
            self.data = [line.strip() for line in f]

    def calculate_information(self): 
        """
        Calculate the information of the number of spins n, the number of states the data can observe 2^n, 
        the number N of datapoints, and the number N_max of different states that can be observed. 
        """
        # Calculate n, the number of spins
        n_values = []
        for i, entry in enumerate(self.data):
            n = sum(digit.isdigit() for digit in entry)
            n_values.append(n)
        
        if len(set(n_values)) == 1:
            print(f"n = {n_values[0]}")
        else: 
            print("n values are not equal")

        # Calculate the number of states that can be observed 
        states = 2**n_values[0]
        print(f"Number of states = {states}")
    
        # Calculate N, the number of datapoints
        N = len(self.data)
        print(f"N = {N}")

        # Calculate N_max, the number of different states that can be observed
        N_max = len(set(self.data))
        print(f"N_max = {N_max}")
    
    def plot_votes(self): 
        """
        Plot the votes of the Supreme Court.
        """

        # Plot the value of <si>_D (the average of the votes) as a function of i. 

        with open(self.SC_file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        data = np.array([[int(char) for char in line] for line in 
                         lines])
        data[data == 0] = -1

        mean_si = np.mean(data, axis=0)
        corr_si_sj = np.einsum('ni,nj->ij', data, data) / data.shape[0]

        sorted_indices = np.argsort(mean_si)
        sorted_mean_si = mean_si[sorted_indices]
        i = np.arange(len(sorted_mean_si))

        plt.figure()
        plt.scatter(i, sorted_mean_si, marker='o', color='black')
        plt.title(r"Mean $<s_i>_D$ as a function of i", fontsize=14)
        plt.xlabel("i", fontsize=12)
        plt.xticks([])
        plt.ylabel(r"$<s_i>_D$", fontsize=12)
        plt.grid(visible=True, alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Heatmap of the correlation matrix
        sorted_corr_si = corr_si_sj[sorted_indices][:, sorted_indices]
        plt.figure()
        plt.imshow(sorted_corr_si, cmap='gray_r')  # Use 'gray_r' for reversed greyscale
        plt.colorbar()
        plt.title(r"Correlation matrix $<s_i s_j>_D$", fontsize=14)
        plt.xlabel("i", fontsize=12)
        plt.ylabel("j", fontsize=12)
        plt.xticks([])
        plt.yticks([])
        plt.grid(visible=True, alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sc = SupremeCourt("data/hi_ussc_unsorted.txt", "data/Jij_ussc_unsorted.txt", "data/US_SupremeCourt.txt")
    sc.load_data()
    sc.calculate_information()
    sc.plot_votes()

    

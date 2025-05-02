import numpy as np 
import matplotlib.pyplot as plt
import os

class SupremeCourt: 
    def __init__(self, num_judges, hi_file_path, Jij_file_path, SC_file_path): 
        """
        Initialize a SupremeCourt object.
        """
        self.hi_file_path = hi_file_path 
        self.Jij_file_path = Jij_file_path 
        self.SC_file_path = SC_file_path 
        self.num_judges = num_judges
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
    
    def plot_average_votes(self, save_path=None): 
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
        if save_path: 
            plt.savefig(save_path)
        else: 
            plt.show()

    def plot_vote_correlation(self, save_path=None): 
        """
        Plot the correlation matrix of the votes.
        """
        # Heatmap of the correlation matrix
        with open(self.SC_file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        data = np.array([[int(char) for char in line] for line in 
                         lines])
        data[data == 0] = -1
        mean_si = np.mean(data, axis=0)
        sorted_indices = np.argsort(mean_si)
        corr_si_sj = np.einsum('ni,nj->ij', data, data) / data.shape[0]
        sorted_corr_si = corr_si_sj[sorted_indices][:, sorted_indices]
        plt.figure()
        plt.imshow(sorted_corr_si, cmap='gray_r')  # Use 'gray_r' for reversed greyscale
        plt.colorbar()
        plt.title(r"Correlation matrix $<s_i s_j>_D$", fontsize=14)
        plt.xlabel("i", fontsize=12)
        plt.ylabel("j", fontsize=12)
        plt.xticks(ticks=np.arange(len(sorted_indices)), labels=sorted_indices, fontsize=8, rotation=90)
        plt.yticks(ticks=np.arange(len(sorted_indices)), labels=sorted_indices, fontsize=8)
        plt.grid(visible=True, alpha=0.5)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path)
        else: 
            plt.show()

    def plot_fitted_parameters(self, save_hi_path=None, save_Jij_path=None): 
        """
        Plot heatmaps of the fitted hi vector and Jij matrix, reordered by <si>_D.
        """
        hi_unsorted = np.loadtxt(self.hi_file_path)
        Jij_unsorted = np.loadtxt(self.Jij_file_path)


        Jij_matrix = np.zeros((self.num_judges, self.num_judges))
        idx = 0
        for i in range(self.num_judges):
            for j in range(i+1, self.num_judges):
                Jij_matrix[i, j] = Jij_unsorted[idx]
                Jij_matrix[j, i] = Jij_unsorted[idx]
                idx += 1

        with open(self.SC_file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Sort by mean magnetisations 
        data = np.array([[int(char) for char in line] for line in lines])
        data[data == 0] = -1
        mean_si = np.mean(data, axis=0)
        sorted_indices = np.argsort(mean_si)

        hi_sorted = hi_unsorted[sorted_indices]
        Jij_sorted = Jij_matrix[sorted_indices][:, sorted_indices]

        # Plot hi heatmap
        plt.figure()
        plt.imshow(hi_sorted.reshape(1, -1), cmap='gray_r', aspect='auto')
        plt.colorbar()
        plt.title(r"Fitted $h_i$ Values", fontsize=14)
        plt.yticks([])
        plt.xticks(ticks=np.arange(len(hi_sorted)), labels=sorted_indices, fontsize=8, rotation=90)
        plt.tight_layout()
        if save_hi_path:
            plt.savefig(save_hi_path)
        else:
            plt.show()

        # Plot Jij heatmap
        plt.figure()
        plt.imshow(Jij_sorted, cmap='gray_r')
        plt.colorbar()
        plt.title(r"Fitted $J_{ij}$ Values", fontsize=14)
        plt.xticks(ticks=np.arange(len(Jij_sorted)), labels=sorted_indices, fontsize=8, rotation=90)
        plt.yticks(ticks=np.arange(len(Jij_sorted)), labels=sorted_indices, fontsize=8)
        plt.tight_layout()
        if save_Jij_path:
            plt.savefig(save_Jij_path)
        else:
            plt.show()


if __name__ == "__main__":
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    sc = SupremeCourt(
        num_judges=9, 
        hi_file_path="data/hi_ussc_unsorted.txt", 
        Jij_file_path="data/Jij_ussc_unsorted.txt", 
        SC_file_path="data/US_SupremeCourt.txt"
        )
    
    sc.load_data()
    sc.calculate_information()
   
    sc.plot_average_votes(save_path=os.path.join(results_dir, "avg_votes_plot.png"))
    sc.plot_vote_correlation(save_path=os.path.join(results_dir, "correlation_heatmap.png"))
    sc.plot_fitted_parameters(
        save_hi_path=os.path.join(results_dir, "hi_heatmap.png"),
        save_Jij_path=os.path.join(results_dir, "Jij_heatmap.png")
    )

    

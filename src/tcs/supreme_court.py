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
        self.hi_unsorted = None
        self.Jij_unsorted = None
        self.data = None
        self.n = None
        self.mean_si_empirical = None
        self.correlation_sisj_empirical = None
        self.mean_si_model = None
        self.correlation_sisj_model = None
        self.p_g = None
        self.p_D = None
        self.all_states = None

    def load_data(self):
        with open(self.SC_file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.data = np.array([[int(c) if c != '0' else -1 for c in line] for line in lines])
        self.n = self.data.shape[1]
    
    def load_parameters(self):
        self.hi_unsorted = np.loadtxt(self.hi_file_path)
        self.Jij_unsorted = np.loadtxt(self.Jij_file_path)

        self.Jij_matrix = np.zeros((self.num_judges, self.num_judges))
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.Jij_matrix[i, j] = self.Jij_unsorted[idx]
                self.Jij_matrix[j, i] = self.Jij_unsorted[idx]
                idx += 1
    
    def calculate_empirical_values(self):
        """
        Calculates <si>D and <si sj>D from the data.
        """
        # Calculate <si>_D and <si sj>_D
        self.mean_si_empirical = np.mean(self.data, axis=0)
        self.correlation_sisj_empirical = np.einsum('ni,nj->ij', self.data, self.data) / self.data.shape[0]
        _, counts = np.unique(self.data, axis=0, return_counts=True)

        # Calculate p_D(s)
        self.p_D = counts / len(self.data)
    
    def calculate_model_values(self):
        """
        Calculates <si>D and <si sj>D from the model.
        """
        states, _ = np.unique(self.data, axis=0, return_counts=True)

        # Calculate p_g(s) 
        if self.p_g is None:
            exp_terms = []
            for state in states:
                exp_term = np.exp(np.dot(self.hi_unsorted, state) + 0.5*np.sum(self.Jij_matrix * np.outer(state, state)))
                exp_terms.append(exp_term)
            exp_terms = np.array(exp_terms)

            Z_g = np.sum(exp_terms)
            self.p_g = exp_terms / Z_g

        # Calculate <si> and <si sj>
        if self.mean_si_model is None or self.correlation_sisj_model is None:
            self.mean_si_model = np.sum(states * self.p_g[:, None], axis=0)
            self.correlation_sisj_model = np.einsum('si,sj,s->ij', states, states, self.p_g)


    def calculate_information(self): 
        """
        Calculate the information of the number of spins n, the number of states the data can observe 2^n, 
        the number N of datapoints, and the number N_max of different states that can be observed. 
        """
        self.calculate_empirical_values()
        # Calculate n, the number of spins
        unique_states = set(map(tuple, self.data))
        
        print(f"n = {self.n}")
        print(f"Number of possible states = {2 ** self.n}")
        print(f"N = {len(self.data)}")
        print(f"N_max = {len(unique_states)}")
    

    def plot_average_votes(self, save_path=None): 
        """
        Plot the votes of the Supreme Court.
        """
        # Plot the value of <si>_D (the average of the votes) as a function of i. 
        self.calculate_empirical_values()
        sorted_indices = np.argsort(self.mean_si_empirical)
        plt.figure()
        plt.scatter(np.arange(self.n), self.mean_si_empirical[sorted_indices], color='black')
        plt.title(r"Mean $<s_i>$ as a function of i")
        plt.xlabel("Justice (sorted)")
        plt.ylabel(r"$<s_i>$")
        plt.grid(True, alpha=0.5)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_vote_correlation(self, save_path=None): 
        """
        Plot a heatmap of the correlation matrix of the Judges' votes.
        """
        self.calculate_empirical_values()
        sorted_indices = np.argsort(self.mean_si_empirical)
        sorted_corr = self.correlation_sisj_empirical[sorted_indices][:, sorted_indices]

        plt.figure()
        plt.imshow(sorted_corr, cmap='gray_r')
        plt.colorbar()
        plt.title(r"Correlation matrix $<s_i s_j>$")
        plt.xticks(np.arange(self.n), sorted_indices, fontsize=8, rotation=90)
        plt.yticks(np.arange(self.n), sorted_indices, fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_fitted_parameters(self, save_hi_path=None, save_Jij_path=None): 
        """
        Plot heatmaps of the fitted hi vector and Jij matrix, reordered by <si>_D.
        """
        self.load_parameters()
        self.calculate_empirical_values()
        sorted_indices = np.argsort(self.mean_si_empirical)

        hi_sorted = self.hi_unsorted[sorted_indices]
        Jij_sorted = self.Jij_matrix[sorted_indices][:, sorted_indices]

        # Plot hi heatmap
        plt.figure()
        plt.imshow(hi_sorted.reshape(1, -1), cmap='gray_r', aspect='auto')
        plt.colorbar()
        plt.title(r"Fitted $h_i$ Values", fontsize=14)
        plt.xlabel("i", fontsize=12)
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
        plt.xlabel("i", fontsize=12)
        plt.ylabel("j", fontsize=12)
        plt.xticks(ticks=np.arange(len(Jij_sorted)), labels=sorted_indices, fontsize=8, rotation=90)
        plt.yticks(ticks=np.arange(len(Jij_sorted)), labels=sorted_indices, fontsize=8)
        plt.tight_layout()

        if save_Jij_path:
            plt.savefig(save_Jij_path)
        else:
            plt.show()

    def plot_probability_cross_validation(self, save_path=None): 
        """
        Plot the probability of pg(s) against pD(s) to cross-validate 
        the theoretical model against the data. 
        """
        self.calculate_empirical_values()
        self.calculate_model_values()

        # Calculate R^2 value
        residuals = self.p_g - self.p_D
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.p_D - np.mean(self.p_D))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Plot the probability of pg(s) against pD(s)
        plt.figure()
        plt.scatter(self.p_D, self.p_g, marker='o', color='black')
        plt.plot(
            [min(self.p_D), max(self.p_D)],
            [min(self.p_D), max(self.p_D)], 
            color='red', 
            label=f'Ideal line: y = x\n$R^2$ = {r_squared:.2f}'
            )
        
        plt.title(r"$p_D(s)$ against $p_g(s)$", fontsize=14)
        plt.xlabel(r"$p_D(s)$", fontsize=12)
        plt.ylabel(r"$p_g(s)$", fontsize=12)
        plt.grid(visible=True, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if save_path: 
            plt.savefig(save_path)
        else: 
            plt.show()

    def plot_averages_cross_validation(self, avg_save_path=None, corr_save_path=None): 
        """
        Plot the average cross-validation results.
        """
        self.calculate_empirical_values()
        self.calculate_model_values()

        # Plot the average <si>_D against <si>
        plt.figure()
        plt.scatter(self.mean_si_empirical, self.mean_si_model, marker='o', color='black')
        residuals = self.mean_si_model - self.mean_si_empirical
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.mean_si_empirical - np.mean(self.mean_si_empirical))**2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.plot(
            [min(self.mean_si_empirical), max(self.mean_si_empirical)], 
            [min(self.mean_si_empirical), max(self.mean_si_empirical)], 
            color='red', 
            label=f'Ideal line: y = x\n$R^2$ = {r_squared:.2f}'
            )
        plt.title(r"$<s_i>_D$ against $<s_i>$", fontsize=14)
        plt.xlabel(r"$<s_i>_D$", fontsize=12)
        plt.ylabel(r"$<s_i>$", fontsize=12)
        plt.grid(visible=True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        if avg_save_path: 
            plt.savefig(avg_save_path)
        else: 
            plt.show()

        # Plot the correlations <si sj>_D against <si sj>
        plt.figure()
        plt.scatter(
            self.correlation_sisj_empirical.flatten(), 
            self.correlation_sisj_model.flatten(), 
            marker='o', 
            color='black'
            )
        residuals = self.correlation_sisj_model.flatten() - self.correlation_sisj_empirical.flatten()
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.correlation_sisj_empirical.flatten() - np.mean(self.correlation_sisj_empirical.flatten()))**2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.plot(
            [min(self.correlation_sisj_empirical.flatten()), max(self.correlation_sisj_empirical.flatten())], 
            [min(self.correlation_sisj_empirical.flatten()), max(self.correlation_sisj_empirical.flatten())], 
            color='red', 
            label=f'Ideal line: y = x\n$R^2$ = {r_squared:.2f}'
            )
        plt.title(r"$<s_i s_j>_D$ against $<s_i s_j>$", fontsize=14)
        plt.xlabel(r"$<s_i s_j>_D$", fontsize=12)
        plt.ylabel(r"$<s_i s_j>$", fontsize=12)
        plt.grid(visible=True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        if corr_save_path: 
            plt.savefig(corr_save_path)
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

    sc.plot_probability_cross_validation(save_path=os.path.join(results_dir, "cross_validation_plot.png"))
    sc.plot_averages_cross_validation(
        avg_save_path=os.path.join(results_dir, "avg_cross_validation_plot.png"),
        corr_save_path=os.path.join(results_dir, "corr_cross_validation_plot.png")
    )

    

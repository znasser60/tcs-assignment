from tcs import Neuron, SupremeCourt
import os

def main(): 
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    neuron = Neuron("data/Data_neuron.txt")
    neuron.load_data()
    neuron.plot_time_distribution(save_path=os.path.join(results_dir, "time_distribution.png"))
    print("Refractory period: ", neuron.calc_refractory_period())
    neuron.fit_exponential(save_path=os.path.join(results_dir, "exponential_fit.png"))
    neuron.generate_spike_data(save_path=os.path.join(results_dir, "generated_data.png"))

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

    sc.plot_conservative_probabilities(save_path=os.path.join(results_dir, "conservative_votes.png"))
    

if __name__ == "__main__":  
    main()

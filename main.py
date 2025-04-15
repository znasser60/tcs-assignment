from tcs import Neuron
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
    # new_data = neuron.generate_spike_data()
    # print("Generated Spike Data: ", new_data)

if __name__ == "__main__":  
    main()

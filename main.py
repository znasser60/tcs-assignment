from tcs import Neuron

def main(): 
    neuron = Neuron("data/Data_neuron.txt")
    neuron.load_data()
    neuron.plot_time_distribution()
    print("Refractory period: ", neuron.calc_refractory_period())
    neuron.fit_exponential()
    new_data = neuron.generate_spike_data()
    print("Generated Spike Data: ", new_data)

if __name__ == "__main__":  
    main()

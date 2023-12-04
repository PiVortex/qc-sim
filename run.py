from lib import Simulation

sim = Simulation(n_excited = 4, n_sites = 11) # Create a new simulation 

for i in range(0, 51):
    print(i, end='    ') 
    probability = sim.apply_obs()
    print(probability)
    sim.apply_cycle()
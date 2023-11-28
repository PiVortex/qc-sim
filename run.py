from lib import Simulation

sim = Simulation(n_excited = 3, n_sites = 8, debug = True) # Create a new simulation 

for i in range(0, 51):
    print(i, end='    ') 
    probability = sim.apply_obs()
    print(probability)
    sim.apply_cycle()


from lib import Simulation

sim = Simulation(n_excited = 1, n_sites = 12) # Create a new simulation 
print(sim.get_current_state())
sim.apply_cycle()
state = sim.get_current_state()
print(state[:, None])
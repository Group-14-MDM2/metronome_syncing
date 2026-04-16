import numpy as np
import matplotlib.pyplot as plt
from mechanical_system.mechanical_lib.mechanical_system import mechanical_system, model_params

def generate_oscillators(N: int, seed: float = 42) -> tuple[np.ndarray, np.ndarray]:
   '''generates some oscillators according to a distribution'''
   np.random.seed(seed)
   lengths = np.array(N*[1])
   initial_angles = np.random.uniform(-2, 2, N)
   initial_conditions = np.vstack((initial_angles, N*[0]))
   return initial_conditions, lengths

def main():
   N = 3
   seed = 41
   initial_conditions, lengths = generate_oscillators(N, seed)

   params = model_params(0.5, 10, lengths, 9.81, -0.1)
   
   simulation = mechanical_system(params, initial_conditions)

   simulation.RK4((0, 20), 8000, 0.8, False)

   simulation.plot_phase_domain()
   simulation.plot_time_domain()
   simulation.plot_order("r")

if __name__ == "__main__":
   main()
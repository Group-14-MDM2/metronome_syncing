import numpy as np
import matplotlib.pyplot as plt
from mechanical_system import mechanical_system, model_params

def generate_oscillators(N: int) -> tuple[np.ndarray, np.ndarray]:
   '''generates some oscillators according to a distribution'''
   lengths = np.random.uniform(0.5, 1.5, N)
   initial_angles = np.random.uniform(-2, 2, N)
   initial_conditions = np.vstack((initial_angles, N*[0]))
   return initial_conditions, lengths

def main():
   N = 100
   initial_conditions, lengths = generate_oscillators(N)

   params = model_params(0.12, 3, lengths, 9.81, 1)
   
   simulation = mechanical_system(params, initial_conditions)

   simulation.RK4((0, 600), 10000)

   simulation.plot_phase_domain()
   simulation.plot_time_domain()
   simulation.plot_order("b")

if __name__ == "__main__":
   main()
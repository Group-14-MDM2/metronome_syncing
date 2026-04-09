import numpy as np
import matplotlib.pyplot as plt
from mechanical_system import mechanical_system, model_params

def main():
   params = model_params(0.12, 3, np.array([1, 1, 1]), 9.81, 0.1)

   initial_conditions = np.array([[1.4, 0.1, 0.25],
                                  [0, 0, 0]])
   
   simulation = mechanical_system(params, initial_conditions)

   simulation.RK4((0, 100), 1000)

   simulation.plot_phase_domain()
   simulation.plot_time_domain()
   simulation.plot_order("b")

if __name__ == "__main__":
   main()
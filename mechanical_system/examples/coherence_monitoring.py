import numpy as np
import matplotlib.pyplot as plt
from mechanical_lib.mechanical_system import mechanical_system, model_params
from mechanical_lib.batch_run import mechanical_sys_batchrunner

def generate_oscillators(N: int, seed: float = 42) -> tuple[np.ndarray, np.ndarray]:
   '''generates some oscillators according to a distribution'''
   np.random.seed(seed)
   lengths = np.array(N*[1])
   initial_angles = np.random.uniform(-2, 2, N)
   initial_conditions = np.vstack((initial_angles, N*[0]))
   return initial_conditions, lengths

def main():

   # the maximum number of metronomes to investigate
   N = 10
   seed = 41


   # builds a list of initial conditions for each number of metronomes
   initial_condition_list = []
   params_list = []
   for i in range(1, N+1):
      initial_conditions, lengths = generate_oscillators(i, seed)
      initial_condition_list.append(initial_conditions)
      params = model_params(1, 10, lengths, 9.81, -0.1)
      params_list.append(params)

   
   # Initialises the batch runner
   batch_runner = mechanical_sys_batchrunner(params_list, initial_condition_list, 0, 30, 600)
   
   # runs the batch runner for all the parameters, finds the time at which they become coherent
   batch_runner.batch_run()
   batch_runner.get_coherence_times()

   # plots the time to get coherent vs the number of oscillators
   fig, ax = plt.subplots()
   ax.plot(range(1, N+1), batch_runner.coherence_times)
   ax.set_xlabel("Number of Oscillators")
   ax.set_ylabel("Time to become coherence (s)")
   plt.show()


if __name__ == "__main__":
   main()
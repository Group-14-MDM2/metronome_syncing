import numpy as np
from kuramoto_simulation import Solver, Window, Screen_params, Model_params, Data_Collector, Standard_Step
from matplotlib import pyplot as plt
from typing import Any


# a data collector that gets all the times and oscillator phases to be plotted later
class Collector(Data_Collector):
   def __init__(self) -> None:
      super().__init__()
   def start(self, solver: Solver, window: Window):
      self.phases = np.array([oscillator.theta for oscillator in window.oscillators])
      self.time = [solver.t]
   def collect(self, solver: Solver, window: Window):
      self.time.append(solver.t)
      new_phases = np.array([oscillator.theta for oscillator in window.oscillators])
      self.phases = np.vstack((self.phases, new_phases))
   def get_data(self) -> tuple[list[float | Any], np.ndarray]:
      return self.time, np.sin(self.phases)

def generate_initial_angles(num_angles: int) -> list[float]:
   return np.linspace(0, 2*np.pi, num=num_angles).tolist()

def generate_natural_frequencies(num_frequencies: int) -> list[float]:
   return np.random.uniform(0, 0.5, num_frequencies).tolist()

def noisy_step(t: float, Y: list[float] | np.ndarray, K: float, N: int, nat_freqs: list[float]) -> np.ndarray:
   dYdt = []
   var = 0.5
   for i in range(N):
      d_theta_dt = nat_freqs[i] + np.random.normal(0, var) + K/N * sum(np.sin(Y[j] - Y[i]) for j in range(N))
      dYdt.append(d_theta_dt)
   return np.array(dYdt)


def main() -> None:
   N = 10
   collector = Collector()
   screen_params = Screen_params(width=800, 
                                 height=800, 
                                 radius=350, 
                                 background_colour=(0, 20, 80))
   
   model_params = Model_params(K=0.4, 
                               natural_frequencies=generate_natural_frequencies(N), 
                               initial_angles=generate_initial_angles(N), 
                               step_function=noisy_step)
   simulation = Window(screen_params, 
                          model_params,
                          collector)
   simulation.main()
   time, phases = collector.get_data()

   num_times = phases.shape[0]
   times = np.linspace(0, time[-1], num_times)

   fig, ax = plt.subplots()
   ax.plot(times, phases)
   ax.set_xlabel("Time (s)")
   ax.set_ylabel("Sine(phase)")
   plt.show()

if __name__ == "__main__":
   main()
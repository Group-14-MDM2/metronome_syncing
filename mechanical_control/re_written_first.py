# Linearised Control

from mechanical_system.mechanical_lib.mechanical_system import model_params, mechanical_system
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Self


def generate_oscillators(N: int, seed: float = 42) -> tuple[np.ndarray, np.ndarray]:
   '''generates some oscillators according to a distribution'''
   np.random.seed(seed)
   lengths = np.array(N*[1])
   initial_angles = np.random.uniform(-2, 2, N)
   initial_conditions = np.vstack((initial_angles, N*[0]))
   return initial_conditions, lengths

def tau(self, q: np.ndarray, dqdt: np.ndarray) -> np.ndarray:
   
   #some data for the control model
   Kp = 50.0      
   Kd = 20.0      
   A = 0.05       
   omega = 5.7 

   t = self.times[-1]

   t_vec = np.zeros(self.n)
   
   # cart position
   x = q[-1]
   x_dot = dqdt[-1]
   
   x_d = A * np.sin(omega * t)
   
   error = x_d - x
   cart_force = Kp * error - Kd * x_dot
   
   t_vec[-1] = cart_force
   
   return t_vec

def main():
   N = 4
   seed = 41
   initial_conditions, lengths = generate_oscillators(N, seed)

   params = model_params(0.25, 
                           3, 
                           np.array([0.1, 0.3, 0.1, 1]), 
                           9.81, 
                           0.01)
         
   simulation = mechanical_system(params, initial_conditions, tau)

   simulation.RK4((0, 15), 1000)

   simulation.plot_phase_domain()
   simulation.plot_time_domain()
   simulation.plot_order("b")

if __name__ == "__main__":
   main()
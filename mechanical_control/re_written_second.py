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
   t = np.zeros(self.n)
   
   # ADDED CONTROL PART START
   
# 1. PD controller gains (tune to adjust sync speed)
   Kp_metronome = 5.0  # Proportional gain for metronomes
   Kd_metronome = 2.0  # Derivative gain for metronomes
   Kp_cart = 2.0       # Proportional gain for cart
   Kd_cart = 0.5       # Derivative gain for cart

   # 2. Calculate virtual control input v (desired acceleration)
   v = np.zeros(self.n)
   q1 = q[0]           # Angle of the leader metronome
   dq1 = dqdt[0]       # Angular velocity of the leader metronome

   # Followers (n-1 metronomes) track the leader (q1)
   for i in range(self.n - 1):
       v[i] = Kp_metronome * (q1 - q[i]) + Kd_metronome * (dq1 - dqdt[i])

   # The cart (last state) tracks a scaled inverse of q1
   scale = 0.05 
   v[-1] = Kp_cart * (-scale * q1 - q[-1]) + Kd_cart * (-scale * dq1 - dqdt[-1])

   # 3. Feedback linearization: Extract cart dynamics (last row)
   M_last_row = self.M(q)[-1, :]
   C_last_row = self.C(q, dqdt)[-1, :]
   G_last_item = self.G(q)[-1] 

   # 4. Compute actual motor torque: tau = M*v + C*q_dot + G
   tau_cart = np.dot(M_last_row, v) + np.dot(C_last_row, dqdt) + G_last_item

   # 5. Apply the control torque only to the cart
   t[-1] = tau_cart
   
   return t

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
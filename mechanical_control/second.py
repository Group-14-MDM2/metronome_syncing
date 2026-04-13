import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable

# setting the model parameters
@dataclass
class model_params:
   n: int
   m: float
   M: float
   lengths: np.ndarray
   initial_theta: np.ndarray
   g: float
   epsilon: float

params = model_params(4,
                     0.25, 
                     3, 
                     np.array([0.1, 0.3, 0.1, 1]), 
                     np.array([0.9, 0.25, 0.35, 0.71]), 
                     9.81, 
                     0.01)

# making all the matricies

def M(q: np.ndarray) -> np.ndarray:
   m = np.zeros((params.n, params.n))

   # the last row is different
   for i in range(params.n - 1):
      for j in range(params.n - 1):
         if i == j:
            m[i, j] = params.m * (params.lengths[i]**2)
      m[i, -1] = params.m * params.lengths[i] * np.cos(q[i])
   
   for j in range(params.n - 1):
      m[-1, j] = params.m * params.lengths[j]*np.cos(q[j])
   m[-1, -1] = params.M + params.n * params.m
   return m

def C(q: np.ndarray, dqdt: np.ndarray) -> np.ndarray:
   c = np.zeros((params.n, params.n))
   
   # the last row is different
   for i in range(params.n - 1):
      for j in range(params.n - 1):
         if i == j:
            c[i, j] = params.m * params.lengths[j]**2 * params.epsilon * ((q[j]/params.initial_theta[j])**2 - 1)
   for j in range(params.n - 1):
      c[-1, j] = params.m * params.lengths[j] * dqdt[j] * np.sin(q[j])
   return c

def G(q: np.ndarray) -> np.ndarray:
   g = np.zeros(params.n)
   for j in range(params.n - 1):
      g[j] = params.m * params.g * params.lengths[j] * np.sin(q[j])
   return g

def tau(q: np.ndarray, dqdt: np.ndarray) -> np.ndarray:
   t = np.zeros(params.n)
   
   # ADDED CONTROL PART START
   
# 1. PD controller gains (tune to adjust sync speed)
   Kp_metronome = 5.0  # Proportional gain for metronomes
   Kd_metronome = 2.0  # Derivative gain for metronomes
   Kp_cart = 2.0       # Proportional gain for cart
   Kd_cart = 0.5       # Derivative gain for cart

   # 2. Calculate virtual control input v (desired acceleration)
   v = np.zeros(params.n)
   q1 = q[0]           # Angle of the leader metronome
   dq1 = dqdt[0]       # Angular velocity of the leader metronome

   # Followers (n-1 metronomes) track the leader (q1)
   for i in range(params.n - 1):
       v[i] = Kp_metronome * (q1 - q[i]) + Kd_metronome * (dq1 - dqdt[i])

   # The cart (last state) tracks a scaled inverse of q1
   scale = 0.05 
   v[-1] = Kp_cart * (-scale * q1 - q[-1]) + Kd_cart * (-scale * dq1 - dqdt[-1])

   # 3. Feedback linearization: Extract cart dynamics (last row)
   M_last_row = M(q)[-1, :]
   C_last_row = C(q, dqdt)[-1, :]
   G_last_item = G(q)[-1] 

   # 4. Compute actual motor torque: tau = M*v + C*q_dot + G
   tau_cart = np.dot(M_last_row, v) + np.dot(C_last_row, dqdt) + G_last_item

   # 5. Apply the control torque only to the cart
   t[-1] = tau_cart
   
   return t


# the time-step function

def step(t: float, s: np.ndarray) -> np.ndarray:
   q = s[0, :]
   dqdt = s[1, :]
   dqqdtt = np.linalg.inv(M(q)) @ (tau(q, dqdt) - C(q, dqdt) @ dqdt - G(q))
   return np.vstack([dqdt, dqqdtt])

# using RK4 to find the solution numerically
def RK4(t_span: tuple[float, float], 
        n: int, 
        y_0: np.ndarray, 
        step: Callable[[float, np.ndarray], np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
   t0 = t_span[0]
   tf = t_span[1]
   h = (tf - t0) / n
   times = [t0]
   Y = [y_0]
   for _ in range(n):
      t = times[-1]
      y_k = Y[-1]
      m1 = step(t, y_k)
      m2 = step(t + h/2, y_k + m1 * h/2)
      m3 = step(t + h/2, y_k + m2 * h/2)
      m4 = step(t + h, y_k + m3 * h)
      y_kp1 = y_k + h/6 * (m1 + 2*m2 + 2*m3 + m4)
      Y.append(y_kp1)
      times.append(t + h)
   return Y, times


def main():
   t0 = 0
   tf = 15
   n = 10000
   y_0 = np.vstack([[0.9, 0.25, 0.35, 0],
                    [0, 0, 0, 0]])
   Y, times = RK4((t0, tf), n, y_0, step)

   '''#plotting in the time domain
   phases = [y[0, :-1] for y in Y]
   plt.plot(times, phases)
   plt.show()'''

   # plotting in phase space
   phases = [y[0, :-1] for y in Y]
   velocities = [y[1, :-1] for y in Y]
   plt.plot(phases, velocities)
   plt.show()


if __name__ == "__main__":
   main()
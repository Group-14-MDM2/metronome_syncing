import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Self

# setting the model parameters
@dataclass
class model_params:
   m: float
   M: float
   lengths: np.ndarray
   g: float
   epsilon: float


class mechanical_system:
   def __init__(self, 
                params: model_params, 
                initial_conditions: np.ndarray,
                tau: Callable[[Self, np.ndarray, np.ndarray], np.ndarray] | None = None) -> None:
      
      assert len(params.lengths) == initial_conditions.shape[1], f"mismatch between the number of lengths given and number of initial conditions"
      
      self.n = len(params.lengths) # number of oscillators
      self.params = params # the model parameters
      self.initial_conditions = initial_conditions
      self.Y = [initial_conditions] # the time-series for the state vector
      self.times = []
      self.initial_theta = initial_conditions[0, :]
      self.tau_func = tau

   def M(self, q: np.ndarray) -> np.ndarray:
      m = np.zeros((self.n, self.n))

      # the last row is different
      for i in range(self.n - 1):
         for j in range(self.n - 1):
            if i == j:
               m[i, j] = self.params.m * (self.params.lengths[i]**2)
         m[i, -1] = self.params.m * self.params.lengths[i] * np.cos(q[i])
      
      for j in range(self.n - 1):
         m[-1, j] = self.params.m * self.params.lengths[j]*np.cos(q[j])
      m[-1, -1] =self.params.M + self.n * self.params.m
      return m

   def C(self, q: np.ndarray, dqdt: np.ndarray) -> np.ndarray:
      c = np.zeros((self.n, self.n))
      
      # the last row is different
      for i in range(self.n - 1):
         for j in range(self.n - 1):
            if i == j:
               c[i, j] = self.params.m * self.params.lengths[j]**2 * self.params.epsilon * ((q[j]/self.initial_theta[j])**2 - 1)
      for j in range(self.n - 1):
         c[-1, j] = self.params.m * self.params.lengths[j] * dqdt[j] * np.sin(q[j])
      return c
   
   def G(self, q: np.ndarray) -> np.ndarray:
      g = np.zeros(self.n)
      for j in range(self.n - 1):
         g[j] = self.params.m * self.params.g * self.params.lengths[j] * np.sin(q[j])
      return g
   
   def tau(self, q: np.ndarray, dqdt: np.ndarray) -> np.ndarray:
      if self.tau_func != None:
         return self.tau_func(self, q, dqdt)
      return np.zeros(self.n)
   
   def step(self, t: float, s: np.ndarray) -> np.ndarray:
      q = s[0, :]
      dqdt = s[1, :]
      ddqdtt = np.linalg.inv(self.M(q)) @ (self.tau(q, dqdt) - self.C(q, dqdt) @ dqdt - self.G(q))
      return np.vstack([dqdt, ddqdtt])
   
   def get_order(self, s: np.ndarray) -> complex:
      '''Finds the order parameter, a complex number'''
      q = s[0, :]
      order = np.sum(np.exp(1j * q)) / self.n
      return order
   
   def moving_average(self, window_size: int) -> None:
      '''Calculates the moving average of the order parameter'''

      self.average_orders = []

      for (index, order) in enumerate(self.orders):
         reference_index = max(index - window_size, 0)
         self.average_orders.append(np.mean(self.orders[reference_index:index]))    
   
   def RK4(self, 
           t_span: tuple[float, float], 
           num_steps: int,
           coherence_threshold: float | None = 0.9,
           stop_after_coherence: bool | None = False) -> None:
      '''Runs RK4 for the mechanical system in the time and for the number of steps specified
         If a coherence threshold parameter is given, the run stops after the moving average reaches this'''

      t0 = t_span[0]
      tf = t_span[1]
      h = (tf - t0) / num_steps

      self.times = [t0]
      self.Y = [self.initial_conditions]
      self.orders = [self.get_order(self.Y[0])]

      for _ in range(num_steps):
         t = self.times[-1]
         y_k = self.Y[-1]
         m1 = self.step(t, y_k)
         m2 = self.step(t + h/2, y_k + m1 * h/2)
         m3 = self.step(t + h/2, y_k + m2 * h/2)
         m4 = self.step(t + h, y_k + m3 * h)
         y_kp1 = y_k + h/6 * (m1 + 2*m2 + 2*m3 + m4)
         self.Y.append(y_kp1)
         self.times.append(t + h)
         self.orders.append(self.get_order(y_kp1))
      self.moving_average(num_steps // 4)

      if coherence_threshold != None:

         # finds the first index when the coherence, r, is above the given threshold
         coherence_index = np.argwhere(np.abs(self.average_orders) > coherence_threshold)[0]
      else:
         coherence_index = num_steps - 1
      
      if stop_after_coherence:
         self.Y = self.Y[:coherence_index]
         self.times = self.times[:coherence_index]
         self.orders = self.orders[:coherence_index]
         self.average_orders = self.average_orders[:coherence_index]

   
   def plot_time_domain(self, file_path: str | None = None) -> None:
      phases = [y[0, :-1] for y in self.Y]
      fig, ax = plt.subplots()

      ax.set_xlabel("Time (s)")
      ax.set_ylabel("sin(Phase) (radians)")
      ax.plot(self.times, np.sin(phases))
      if file_path != None:
         plt.savefig(file_path)
      plt.show()
   
   def plot_phase_domain(self, file_path: str | None = None) -> None:
      phases = [y[0, :-1] for y in self.Y]
      velocities = [y[1, :-1] for y in self.Y]
      fig, ax = plt.subplots()

      ax.set_xlabel("Angular Velocity (radian/s)")
      ax.set_ylabel("Phase (radians)")
      ax.plot(velocities, phases)
      if file_path != None:
         plt.savefig(file_path)
      plt.show()
   
   def plot_order(self, style: str, file_path: str | None = None) -> None:
      '''plots the order parameter according to the style'''

      r_list = [np.abs(order) for order in self.orders]
      psi_list = [np.atan(order.imag/order.real) for order in self.orders if order.real != 0]
      mean_r_list = [np.abs(order) for order in self.average_orders]
      mean_psi_list = [np.atan(order.imag/order.real) for order in self.average_orders]

      match style:
         case "r":
            fig, ax = plt.subplots()
            ax.plot(self.times, r_list, label="Coherence")
            ax.plot(self.times, mean_r_list, label="Moving Average")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Coherence, r (Dimensionless)")
            ax.set_ylim(0, 1)
            fig.legend()
            if file_path != None:
               plt.savefig(file_path)
            plt.show()
         case "psi":
            fig, ax = plt.subplots()
            ax.plot(self.times, psi_list, label="Average Phase")
            ax.plot(self.times, mean_psi_list, label="Moving Average Phase")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Mean Phase (Radians)")
            ax.set_ylim(0, 2*np.pi)
            fig.legend()
            if file_path != None:
               plt.savefig(file_path)
            plt.show()
         case "phase_space":
            fig, ax = plt.subplots()
            ax.plot(r_list, psi_list)
            ax.set_xlabel("Coherence, r (Dimensionless)")
            ax.set_ylabel("Mean Phase (Radians)")
            fig.legend()
            if file_path != None:
               plt.savefig(file_path)
            plt.show()
         case "both" | "b":
            fig, ax = plt.subplots(2, layout="constrained")
            ax[0].plot(self.times, psi_list, label="Average_Phase")
            ax[0].plot(self.times, mean_psi_list, label="Moving Average Phase")
            ax[0].set_xlabel("Time (s)")
            ax[0].set_ylabel("Mean Phase (Radians)")
            ax[0].set_ylim(0, 2*np.pi)
            ax[1].plot(self.times, r_list, label="Coherence")
            ax[1].plot(self.times, mean_r_list, label="Moving Average Coherence")
            ax[1].set_xlabel("Time (s)")
            ax[1].set_ylabel("Coherence, r (Dimensionless)")
            ax[1].set_ylim(0, 1)
            fig.legend()
            if file_path != None:
               plt.savefig(file_path)
            plt.show()
         case _:
            print("UNRECOGNISED PARAMETER - Allowed are: 'r', 'psi', 'both' (or 'b'), 'phase_space'")
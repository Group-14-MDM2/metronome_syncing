import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

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

params = model_params(3, 0.25, 3, np.array([0.3, 0.25, 0.35]), np.array([0, 0.31, 0]), 9.81, 0.01)

# making all the matricies

def M(q: np.ndarray) -> np.ndarray:
   m = np.zeros((params.n, params.n))

   # the last row is different
   for i in range(params.n - 1):
      for j in range(params.n - 1):
         if i == j:
            m[i, j] = params.m * (params.lengths[i]**2)
      m[i, -1] = m * params.lengths[i] * np.cos(q[i])
   
   for j in range(params.n - 1):
      m[-1, j] = m * params.lengths[j]*np.cos(q[j])
   m[-1, -1] = params.M + params.n * params.m
   return m

def C(q: np.ndarray, dqdt: np.ndarray) -> np.ndarray:
   c = np.zeros((params.n, params.n))
   
   # the last row is different
   for i in range(params.n - 1):
      for j in range(params.n - 1):
         if i == j:
            c[i, j] = params.m * params.lengths[j]**2 * ((q[j]/params.initial_theta[j])**2 - 1)
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
   return t


# the time-step function

def step(s: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
   q = s[0]
   dqdt = s[1]
   dqqdtt = np.linalg.inv(M(q)) @ (tau(q, dqdt) - C(q, dqdt) @ dqdt - G(q))
   return (dqdt, dqdt)

# using RK4 to find the solution numerically


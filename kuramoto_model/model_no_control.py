import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as svp

def step(t, Y, K, N, omega):
   '''this is the step function for use in a numerical ODE'''
   new_speeds = []
   for i in range(N):
      new_speed = omega[i] + K/N * sum(np.sin(Y[j] - Y[i]) for j in range(N))
      new_speeds += new_speed
   return np.array(new_speeds)

N = 4
time_frame = (0, 100)
initial_conditions = np.random.random(N)
K = 0.2
omega = np.random.random(N)

solution = svp(step, time_frame, initial_conditions, args=(K, N, omega))

fig, ax = plt.subplots()
ax.plot(solution.t, solution.y)
plt.show()
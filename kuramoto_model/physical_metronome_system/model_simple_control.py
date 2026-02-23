import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as svp
import argparse

def control(Y, M):
   result = M * np.sum(np.sin(Y))
   return result

def step(t, Y, K, N, M, omega):
   '''this is the step function for use in a numerical ODE'''
   new_speeds = []
   for i in range(N):
      new_speed = omega[i] + control(Y, M) + (K/N) * sum(np.sin(Y[j] - Y[i]) for j in range(N))
      new_speeds.append(new_speed)
   return np.array(new_speeds)


def main():
   argument_parser = argparse.ArgumentParser()
   argument_parser.add_argument("-num_oscillators", type=int, default=5, help="The number of oscillators to be modelled")
   argument_parser.add_argument("-tf", type=float, default=50.0, help="What time to stop the model at.")
   argument_parser.add_argument("-num_steps", type=int, default=500, help="How many steps the solver runs for")
   argument_parser.add_argument("-coupling_strength", type=float, default=0.4, help="The coupling strength, K")
   argument_parser.add_argument("-control_strength", type=float, default=1, help="How strong the controller is.")
   parsed = argument_parser.parse_args()

   np.random.seed = 42

   N = parsed.num_oscillators
   K = parsed.coupling_strength
   M = parsed.control_strength
   tf = parsed.tf
   num_steps = parsed.num_steps

   time_frame = (0, tf)
   time_span = np.linspace(0, tf, num_steps)
   initial_conditions = np.random.uniform(-1, 1, N)
   omega = np.random.uniform(0.1, 0.3, N)

   solution = svp(step,
                  time_frame, 
                  initial_conditions, 
                  args=(K, N, M, omega),
                  atol=1e-6, rtol=1e-9, 
                  t_eval=time_span)
   
   print(solution)
   fig, ax = plt.subplots()
   for i in range(N):
      ax.plot(solution.t, np.sin(solution.y[i]))
   ax.set_xlabel("Time (s)")
   ax.set_ylabel("Sin(Angle)")
   plt.show()

if __name__ == "__main__":
   main()
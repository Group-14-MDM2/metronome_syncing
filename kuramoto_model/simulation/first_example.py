import numpy as np
from kuramoto_simulation import Window, Screen_params, Model_params

def step(t: float, Y: list[float], K: float, N: int, nat_freqs: list[float]) -> np.ndarray:
   dYdt = []
   for i in range(N):
      d_theta_dt = nat_freqs[i] + K/N * sum(np.sin(Y[j] - Y[i]) for j in range(N))
      dYdt.append(d_theta_dt)
   return np.array(dYdt)


def main():
   screen_params = Screen_params(800, 800, 350, (0, 20, 80))
   model_params = Model_params(0.8, [0.2, 0.4, 0.5, 1], [0.6, 0.8, 0.1, 0], step)
   simulation = Window(screen_params, 
                          model_params)
   simulation.main()

if __name__ == "__main__":
   main()
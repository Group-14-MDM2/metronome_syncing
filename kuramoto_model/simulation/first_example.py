import numpy as np
from kuramoto_simulation import Window, Screen_params, Model_params, Standard_Step

def generate_initial_angles(num_angles: int) -> list[float]:
   return np.linspace(0, 2*np.pi, num=num_angles).tolist()

def generate_natural_frequencies(num_frequencies: int) -> list[float]:
   return np.random.uniform(0, 0.5, num_frequencies).tolist()

def step(t: float, Y: list[float] | np.ndarray, K: float, N: int, nat_freqs: list[float]) -> np.ndarray:
   dYdt = []
   var = 0.5
   for i in range(N):
      d_theta_dt = nat_freqs[i] + np.random.normal(0, var) + K/N * sum(np.sin(Y[j] - Y[i]) for j in range(N))
      dYdt.append(d_theta_dt)
   return np.array(dYdt)


def main() -> None:
   N = 8
   screen_params = Screen_params(width=800, 
                                 height=800, 
                                 radius=350, 
                                 background_colour=(0, 20, 80))
   
   model_params = Model_params(K=1, 
                               natural_frequencies=generate_natural_frequencies(N), 
                               initial_angles=generate_initial_angles(N), 
                               step_function=step)
   simulation = Window(screen_params, 
                          model_params)
   simulation.main()

if __name__ == "__main__":
   main()
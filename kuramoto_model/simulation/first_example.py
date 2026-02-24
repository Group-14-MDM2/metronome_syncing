import numpy as np
from kuramoto_simulation import Window, Screen_params, Model_params, Standard_Step

def generate_initial_angles(num_angles: int) -> list[float]:
   return np.linspace(0, 2*np.pi, num=num_angles).tolist()

def generate_natural_frequencies(num_frequencies: int) -> list[float]:
   return np.random.normal(0, 0.5, num_frequencies).tolist()


def main() -> None:
   N = 8
   screen_params = Screen_params(width=800, 
                                 height=800, 
                                 radius=350, 
                                 background_colour=(0, 20, 80))
   
   model_params = Model_params(K=1, 
                               natural_frequencies=generate_natural_frequencies(N), 
                               initial_angles=generate_initial_angles(N), 
                               step_function=Standard_Step)
   simulation = Window(screen_params, 
                          model_params)
   simulation.main()

if __name__ == "__main__":
   main()
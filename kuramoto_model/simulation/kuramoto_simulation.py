import pygame as pg
import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Any

'''
This is a simple library to render the Kuramoto oscillators.
The idea is that all the code here is for the general case, which will then be imported into a new file to create the simulation.
Basically, don't change this code.
'''

def Standard_Step(t: float, 
         Y: list[float] | np.ndarray, 
         K: float, N: int, 
         nat_freqs: list[float]) -> np.ndarray:
   dYdt = []
   for i in range(N):
      d_theta_dt = nat_freqs[i] + K/N * sum(np.sin(Y[j] - Y[i]) for j in range(N))
      dYdt.append(d_theta_dt)
   return np.array(dYdt)

@dataclass
class Screen_params:
   width: int
   height: int
   radius: int
   background_colour: tuple[int, int, int] = (0, 20, 80)

class Data_Collector:
   '''This is a simple data collector, that is initialised at the start of the simulation.
      At the end of each frame the state of the entire model as well as the solver is collected.
      Using the method get_data() the data can be got out of the model to be used in calculations'''
   def __init__(self) -> None:
      ...
   def start(self, solver: 'Solver', window: 'Window') -> None:
      ...
   def collect(self, solver:'Solver', window: 'Window') -> None:
      ...
   def get_data(self) -> Any:
      ...


class Model_params:
   def __init__(self, K: float,
               natural_frequencies: list[float],
               initial_angles: list[float],
               step_function: Callable[[float, list[float] | np.ndarray, float, int, list[float]], np.ndarray]) -> None:
      self.K: float = K
      self.natural_frequencies = natural_frequencies
      self.initial_angles = initial_angles
      self.step_function = step_function
      self.N = self.get_n()

   def get_n(self) -> int:
      assert len(self.natural_frequencies) == len(self.initial_angles), "The number of natural frequencies doesn't match the number of initial positions"
      return len(self.natural_frequencies)

class Oscillator:
   def __init__(self, nat_frequency: float, initial_pos: float, window) -> None:
      self.omega = nat_frequency
      self.theta = initial_pos
      self.window = window
      self.colour = np.random.randint(100, 255, 3).tolist()

   def draw(self) -> None:
      w = self.window.screen_params.width
      h = self.window.screen_params.height
      r = self.window.screen_params.radius
      screen_pos = (w/2 + r*np.cos(self.theta), h/2 + r*np.sin(self.theta))
      pg.draw.circle(self.window.screen, self.colour, screen_pos, 5)

   def __repr__(self) -> str:
      return f"Natural Frequency: {self.omega}, Angle: {self.theta}"


class Solver:
   def __init__(self, 
                oscillators: list[Oscillator], 
                fun: Callable[[float, list[float] | np.ndarray, float, int, list[float]], np.ndarray], 
                K: float, 
                N: int) -> None:
      '''makes a solver for the problem, using a user defined step function.
      This step function must be of the form f(t, Y, K, N, Omega), where K and N are the coupling strength and number of oscillators.
      When this solver is called it takes as input a time and index and returns '''
      self.oscillators = oscillators
      self.K = K
      self.N = N
      self.rhs = fun
      self.omega = [oscillator.omega for oscillator in oscillators]
      self.t = 0
      self.num_evals = 0
   
   def step(self, dt) -> np.ndarray:
      '''performs one step of RK4 with the delta_t'''
      Y = np.array([oscillator.theta for oscillator in self.oscillators])

      m1 = self.rhs(dt, Y, self.K, self.N, self.omega)
      m2 = self.rhs(self.t + dt/2, Y + dt*m1/2, self.K, self.N, self.omega)
      m3 = self.rhs(self.t + dt/2, Y + dt*m2/2, self.K, self.N, self.omega)
      m4 = self.rhs(self.t + dt, Y + dt*m3, self.K, self.N, self.omega)
      Y += dt/6 * (m1 + 2*m2 + 2*m3 + m4)
      self.t += dt
      self.num_evals += 1
      return Y
   
   def __call__(self, delta) -> np.ndarray:
      return self.step(delta)
   
   def __repr__(self) -> str:
      return f"time elapsed: {self.t}, number of update calls: {self.num_evals}"

class Window:
   def __init__(self, 
               screen_params: Screen_params, 
               model_params: Model_params,
               data_collector: Data_Collector = Data_Collector()) -> None:

      self.screen_params = screen_params
      self.model_params = model_params
      self.data_collector = data_collector
      self.oscillators = []

   def quit(self) -> bool:
      '''First checks if the window is being closed using the button
         Then checks if the 'q' button has been pressed.'''
      
      for event in pg.event.get():
         if event.type == pg.QUIT:
            return True
      keys_pressed = pg.key.get_pressed()
      if keys_pressed[pg.K_q] or keys_pressed[pg.K_ESCAPE]:
         return True
      return False

   def start(self) -> None:
      '''sets up the screen and the numerical solver'''

      # initialises the screen
      pg.init()
      self.screen = pg.display.set_mode((self.screen_params.width, self.screen_params.height))
      self.screen.fill(self.screen_params.background_colour)
      pg.display.flip()

      # initialises the oscillators
      for (omega, theta) in zip(self.model_params.natural_frequencies, self.model_params.initial_angles):
         self.oscillators.append(Oscillator(omega, theta, self))
      
      # initialises the solvers
      self.solver = Solver(self.oscillators, self.model_params.step_function, self.model_params.K, self.model_params.N)
      self.data_collector.start(self.solver, self)
               
   def update(self) -> None:
      prev_time = time.perf_counter()
      while not self.quit():
         
         # clears the screen and fills it in with its background colour
         self.screen.fill(self.screen_params.background_colour)

         # draws the circle the points move around in
         pg.draw.circle(self.screen, (50, 50, 50), 
                        (self.screen_params.width//2, 
                         self.screen_params.height//2),
                         self.screen_params.radius,
                         width=2)
         
         # finds the time since the last frame was drawn
         current_time = time.perf_counter()
         dt = current_time - prev_time

         # updates the solver for the new time and then updates the position of the oscillator
         current_state = self.solver(dt)
         for i, oscillator in enumerate(self.oscillators):
            oscillator.theta = current_state[i]
            oscillator.draw()

         self.data_collector.collect(self.solver, self)
         prev_time = current_time
         pg.display.flip()

   def exit(self) -> None:
      print(self)

   def main(self) -> None:
      self.start()
      self.update()
      self.exit()
   
   def __repr__(self) -> str:
      representation = f'''Coupling Strength: {self.model_params.K},\n\
Number of Oscillators: {self.model_params.N} \n'''
      print("")
      print("Oscillators:")
      for oscillator in self.oscillators:
         print(oscillator)
      print("")
      print(self.solver)
      return representation
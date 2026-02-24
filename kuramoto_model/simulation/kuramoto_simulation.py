import pygame as pg
import numpy as np
from scipy.integrate import solve_ivp as svp
import time
from dataclasses import dataclass
from typing import Callable

'''
This is a simple library to render the Kuramoto oscillators.
The idea is that all the code here is for the general case, which will then be imported into a new file to create the simulation.
Basically, don't change this code.
'''

@dataclass
class Screen_params:
   width: int
   height: int
   radius: int
   background_colour: tuple[int, int, int] = (0, 20, 80)

@dataclass
class Model_params:
   K: float
   natural_frequencies: list[float]
   initial_angles: list[float]
   step_function: Callable[[float, list[float], float, int, list[float]], np.ndarray]

class Oscillator:
   def __init__(self, nat_frequency: float, initial_pos: float, window) -> None:
      self.omega = nat_frequency
      self.theta = initial_pos
      self.window = window
      self.colour = np.random.randint(0, 255, 3).tolist()

   def draw(self) -> None:
      w = self.window.screen_dimensions[0]
      h = self.window.screen_dimensions[1]
      r = self.window.radius
      screen_pos = (w/2 + r*np.cos(self.theta), h/2 + r*np.sin(self.theta))
      pg.draw.circle(self.window.screen, self.colour, screen_pos, 5)

   def __repr__(self) -> str:
      return f"Natural Frequency: {self.omega}, Angle: {self.theta}"


class Solver:
   def __init__(self, oscillators: list[Oscillator], fun, K: float, N: int) -> None:
      '''makes a solver for the problem, using a user defined step function.
      This step function must be of the form f(t, Y, K, N, Omega), where K and N are the coupling strength and number of oscillators.
      When this solver is called it takes as input a time and index and returns '''
      self.oscillators = oscillators
      self.K = K
      self.N = N
      self.rhs = fun
      self.omega = [oscillator.omega for oscillator in oscillators]
      self.t = 0
   
   def step(self, dt):
      '''performs one step of RK4 with the delta_t'''
      Y = np.array([oscillator.theta for oscillator in self.oscillators])

      m1 = self.rhs(dt, Y, self.K, self.N, self.omega)
      m2 = self.rhs(self.t + dt/2, Y + dt*m1/2, self.K, self.N, self.omega)
      m3 = self.rhs(self.t + dt/2, Y + dt*m2/2, self.K, self.N, self.omega)
      m4 = self.rhs(self.t + dt, Y + dt*m3, self.K, self.N, self.omega)
      Y += dt/6 * (m1 + 2*m2 + 2*m3 + m4)
      self.t += dt
      return Y
   
   def __call__(self, delta) -> list[float]:
      return self.step(delta)

class Window:
   def __init__(self, 
               screen_params: Screen_params, 
               model_params: Model_params) -> None:
      
      assert len(model_params.natural_frequencies) == len(model_params.initial_angles), "The number of natural frequencies doesn't match the number of initial positions"

      self.screen_dimensions = (screen_params.width, screen_params.height)
      self.background_colour = screen_params.background_colour
      self.radius = screen_params.radius

      self.N = len(model_params.natural_frequencies)
      self.K = model_params.K
      self.nat_freqs = model_params.natural_frequencies
      self.start_thetas = model_params.initial_angles

      self.oscillators = []
      self.step_function = model_params.step_function

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
      self.screen = pg.display.set_mode(self.screen_dimensions)
      self.screen.fill(self.background_colour)
      pg.display.flip()

      # initialises the oscillators
      for (omega, theta) in zip(self.nat_freqs, self.start_thetas):
         self.oscillators.append(Oscillator(omega, theta, self))
      
      # initialises the solvers
      self.solver = Solver(self.oscillators, self.step_function, self.K, self.N)
               
   def update(self) -> None:
      prev_time = time.perf_counter()
      while not self.quit():
         
         # clears the screen and fills it in with its background colour
         self.screen.fill(self.background_colour)

         # draws the circle the points move around in
         pg.draw.circle(self.screen, (50, 50, 50), 
                        (self.screen_dimensions[0]//2, 
                         self.screen_dimensions[1]//2),
                         self.radius,
                         width=2)
         
         # finds the time since the last frame was drawn
         current_time = time.perf_counter()
         dt = current_time - prev_time

         # updates the solver for the new time and then updates the position of the oscillator
         current_state = self.solver(dt)
         for i, oscillator in enumerate(self.oscillators):
            oscillator.theta = current_state[i]
            oscillator.draw()

         prev_time = current_time
         pg.display.flip()

   def exit(self) -> None:
      print(self)

   def main(self) -> None:
      self.start()
      self.update()
      self.exit()
   
   def __repr__(self) -> str:
      representation = f'''Coupling Strength: {self.K},\n\
Number of Oscillators: {self.N} \n'''
      print("")
      print("Oscillators:")
      for oscillator in self.oscillators:
         print(oscillator)
      print("")
      return representation
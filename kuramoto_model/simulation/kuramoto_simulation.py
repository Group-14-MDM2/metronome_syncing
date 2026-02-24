import pygame as pg
import numpy as np
from scipy.integrate import solve_ivp as svp

'''
This is a simple library to render the Kuramoto oscillators.
The idea is that all the code here is for the general case, which will then be imported into a new file to create the simulation.
Basically, don't change this code.
'''

class Oscillator:
   def __init__(self, nat_frequency, initial_pos) -> None:
      self.omega = nat_frequency
      self.theta = initial_pos

   def __repr__(self) -> str:
      return f"Natural Frequency: {self.omega}, Angle: {self.theta}"


class Solver:
   def __init__(self, oscillators: list[Oscillator], fun, t_span: tuple[float, float], K, N) -> None:
      '''makes a solver for the problem, using a user defined step function.
      This step function must be of the form f(t, Y, K, N), where K and N are the coupling strength and number of oscillators.
      When this solver is called it takes as input a time and index and returns '''
      self.solution = svp(fun, 
                          t_span, 
                          [oscillator.theta for oscillator in oscillators], 
                          dense_output=True,
                          args=(K, N))
   
   def __call__(self, time) -> list[float]:
      return self.solution(time)
   def __repr__(self) -> str:
      return str(self.solution)

class Window:
   def __init__(self, width: int, 
               height: int, 
               background_colour: tuple[int, int, int], 
               coupling_strength: float,
               nat_frequencies: list[float],
               start_positions: list[float],
               t_span: tuple[float, float] = (0, 50),
               delta: float = 0.001) -> None:
      
      assert len(nat_frequencies) == len(start_positions), "The number of natural frequencies doesn't match the number of initial positions"

      self.screen_dimensions = (width, height)
      self.background_colour = background_colour

      self.N = len(nat_frequencies)
      self.K = coupling_strength
      self.nat_freqs = nat_frequencies
      self.start_thetas = start_positions
      self.t_span = t_span

      self.t = t_span[0]
      self.delta = delta
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
      pg.init()
      self.screen = pg.display.set_mode(self.screen_dimensions)
      self.screen.fill(self.background_colour)
      self.solver = Solver(self.oscillators, lambda a, b, c, d: b, self.t_span, self.K, self.N)
      pg.display.flip()

      for (omega, theta) in zip(self.nat_freqs, self.start_thetas):
         self.oscillators.append(Oscillator(omega, theta))
               
   def update(self) -> None:
      while not self.quit() and self.t < self.t_span[1]:
         self.screen.fill(self.background_colour)

         pg.display.flip()
         self.t += self.delta

   def exit(self) -> None:
      ...

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

def main():
   window = Window(500, 500, (0, 15, 80), 0.3, [0.3, 0.2, 0.1], [0.3, 0.2, 0.1])
   window.main()


if __name__ == "__main__":
   main()
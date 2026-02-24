import numpy as np
import pygame as pg
from kuramoto_simulation import Window

def step(t, Y, K, N, nat_freqs):
   dYdt = []
   for i in range(N):
      d_theta_dt = nat_freqs[i] + K/N * sum(np.sin(Y[j] - Y[i]) for j in range(N))
      dYdt.append(d_theta_dt)
   return np.array(dYdt)


def main():
   simulation = Window(500, 500, 100, 
                          (0, 20, 80), 
                          0.4, 
                          [0.2, 0.4, 0.5],
                          [0.6, 0.8, 0.1],
                          step)
   simulation.main()

if __name__ == "__main__":
   main()
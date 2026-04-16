import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Self
from mechanical_lib.mechanical_system import mechanical_system, model_params

class mechanical_sys_batchrunner:
   def __init__(self, params: list[model_params] | model_params, 
                initial_conditions: np.ndarray | list[np.ndarray],
                t0: float,
                tf: float,
                n_steps: int,
                coherence_threshold: float = 0.9) -> None:
      self.params = params
      self.initial_conditions = initial_conditions
      self.t_span = (t0, tf)
      self.n_steps = n_steps
      self.coherence_threshold = coherence_threshold

      # where all the models are stored after being batch run
      self.models = []

      assert self.check_variables() == True

   def check_variables(self) -> bool:
      '''checks whether or not the batch_runner has been initialised properly'''
      if type(self.params) == list and type(self.initial_conditions) == list:
         if len(self.params) == len(self.initial_conditions):
            return True
         print(f"Length of model parameters {len(self.params)} does not match the length of the initial conditions {len(self.initial_conditions)}")
         return False
      else:
         if type(self.params) == model_params and type(self.initial_conditions) == np.ndarray:
            print("Debugger Redundant Here, but continuing")
      return True
   
   def build_and_run(self, params: model_params, initial_conds: np.ndarray):
      '''builds and runs a model based on its parameters'''
      model = mechanical_system(params, initial_conds)
      model.RK4(self.t_span, self.n_steps, self.coherence_threshold)
      self.models.append(model)
   
   def batch_run(self, elementwise: bool = False) -> None:
      '''Does a batch run for the mechanical system
         If elementwise = True => runs for each combination of parameter/initial condition'''
      if type(self.params) == list and type(self.initial_conditions) == list:
         if elementwise:
            for params in self.params:
               for initial_conds in self.initial_conditions:
                  self.build_and_run(params, initial_conds)
         else:
            for i, (params, initial_conds) in enumerate(zip(self.params, self.initial_conditions)):
               self.build_and_run(params, initial_conds)
               if i+1 % 10:
                  print(f"Completed {i+1} runs")
      
      elif type(self.params) == model_params and type(self.initial_conditions) == list:
        for i, initial_conds in enumerate(self.initial_conditions):
           self.build_and_run(self.params, initial_conds)
           if i+1 % 10:
              print(f"Completed {i+1} runs")
      
      elif type(self.params) == list and type(self.initial_conditions) == np.ndarray:
         for i, params in enumerate(self.params):
            self.build_and_run(params, self.initial_conditions)
            if i+1 % 10:
              print(f"Completed {i+1} runs")
      
      elif type(self.params) == model_params and type(self.initial_conditions) == np.ndarray:
         self.build_and_run(self.params, self.initial_conditions)
      else:
         print("Unrecognised combination")
   
   def get_coherence_times(self) -> None:
      '''gets an array of times when the system reaches coherence'''
      self.coherence_times = []
      for model in self.models:
         self.coherence_times.append(model.coherence_time)
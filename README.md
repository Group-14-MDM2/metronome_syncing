## Group 14 - Metronome Syncing

### About this Project
This is Group 14's MDM2 project where we have built, tested and developed code to investigate different control strategies for a system of coupled oscillators,
such as metronomes on a shaky table. In this project we created a small library, mechanical_lib, that implements one of these models. It was designed to be flexible for the different control methods,
allowing the same model to be used without having to validate it each time. 

#### Quick Tour:
  - Environment and pyproject are used to manage the dependencies and modules for the whole project
  - mechanical_lib is where the module that implements the mechanical system and its evaluation lives
  - mechanical_control holds the code that implemented the feedback linearisation and PD control
  - PyTorchRL_mechanical_control holds the code that implements control with Reinforcement Learning
  - mech_control_energy evaluates how much energy the controller models use for given parameters
  - kuramoto_model develops the Kuramoto interpretation of the oscillating system, includes basic model, and adaptive rule to simulate control, with energy/time tradeoffs for 2, and many oscillators.

## Explanation of Contents:

### Kuramoto:
Looks at system evolution according to the Kuramoto model for 2 oscillators, and arbitrarily many oscillators, exploring effect of coupling strength parameter K. Introduces adaptive rule for K to moderate energy spend and synchronise faster, tuned by some arbitrary parameter. Synchronisation time against energy spend plotted for values of this parameter, for a worst case initial setup.

### Mechanical:

pareto_comparison.py - compares PD and FL controllers for synchronising four mechanically coupled metronomes by sweeping across controller parameters and plotting synchronisation time against control energy as a Pareto front,  optimal parameter recommendations are printed in a table for both controllers


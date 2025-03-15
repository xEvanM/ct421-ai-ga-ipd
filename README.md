# README: IPD Simulation with Genetic Algorithm

**Author:** Evan Murphy | **Student ID:** 21306306

## Overview
This project runs a simulation of the Iterated Prisoner's Dilemma (IPD) with a list of fixed strategies and evolves genome based strategies using a genetic algorithm (GA).

## Requirements
- Python 3.x
- Libraries: `numpy`, `matplotlib`

## Installation
1. Install Python 3.x.
2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

# Execution Instructions
1. Save the code in a file, e.g., ipd_ga.py.
    Run the script:
    ```bash
    python ipd_ga.py
    ```

# Default Execution Params
Runs the GA with:
1. Population size: 500
2. Generations: 50
3. Mutation rate: 0.4
4. Crossover rate: 0.6
5. Rounds per game: 100
6. Memory length: 2
7. Opponent: AlwaysCooperate()
8. Noise enabled: True 
9. Noise probability: 0.5

# Output provided
Outputs best strategy genome and fitness progression plot.

# Executing custom runs
Modify parameters in the genetic_algorithm() call in the if __name__ == "__main__": block.
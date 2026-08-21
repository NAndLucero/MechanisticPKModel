# Mechanistic Pharmacokinetic & Dissolution Model

## Overview
This repository contains an Object-Oriented Python application that simulates the dissolution and systemic absorption of a solid oral pharmaceutical dosage form. By modeling the gastrointestinal tract and blood plasma according to a one-compartment model, the script evaluates formulation performance and predicts first-in-human pharmacokinetic profiles.

## Mathematical Fundamentals
The system solves coupled ordinary differential equations (ODEs) to track mass transfer across three states: Solid Drug, GI Tract Fluid, and Blood Plasma.

The physical dissolution of the tablet is modeled using the Noyes-Whitney equation, incorporating the Hixson-Crowell cube root law to account for the shrinking surface area of the solid particle:

$$\frac{dM_{solid}}{dt} = -k_d \cdot M_{solid}^{2/3} \cdot \left( C_s - \frac{M_{GI}}{V_{GI}} \right)$$

The systemic absorption and elimination are modeled as first-order kinetic processes:

$$\frac{dM_{blood}}{dt} = k_a \cdot M_{GI} - k_e \cdot M_{blood}$$

## Features
* **OOP Architecture:** The `MechanisticPKModel` class allows for rapid instantiation of different drug formulations (e.g., fast-dissolving vs. extended-release) by adjusting physiological and physicochemical parameters.
* **Numerical Integration:** Utilizes `scipy.integrate.solve_ivp` (Radau method) to handle the stiff differential equations caused by rapid concentration spikes during initial dissolution.
* **Dosage Optimization:** Includes a numerical optimization wrapper using `scipy.optimize.minimize_scalar` to calculate the exact initial tablet dose required to achieve a specific target maximum blood concentration ($C_{max}$).

## Dependencies
* `numpy`
* `pandas`
* `scipy`
* `matplotlib`

## Usage
Run the script to execute the baseline simulation for a 500mg tablet. The output will generate a matplotlib graph displaying the mass transfer across the three compartments over a 24-hour period.

To find the optimal dosage for a specific therapeutic window, call the optimization function:
```python
# Example: Find the dose required to hit a peak concentration of 8.5 mg/L
ideal_dose = optimize_dosage(target_cmax=8.5)

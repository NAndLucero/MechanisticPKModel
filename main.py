import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


class MechanisticPKModel:
    def __init__(self, dose, k_d, C_s, V_GI, k_a, k_e, V_d):
        """
        Formulation and physiological parameters.
        """
        self.dose = dose  # Initial solid API mass (mg)
        self.k_d = k_d  # Dissolution rate constant
        self.C_s = C_s  # Solubility limit (mg/L)
        self.V_GI = V_GI  # Volume of GI tract (L)
        self.k_a = k_a  # Absorption rate constant (1/hr)
        self.k_e = k_e  # Elimination rate constant (1/hr)
        self.V_d = V_d  # Volume of distribution (L)

    def system_dynamics(self, t, y):
        """
        Coupled ODEs representing the mass balances.
        """
        M_solid, M_GI, M_blood = y

        # --- Prevent negative mass and imaginary numbers during integration ---
        M_solid = max(0, M_solid)

        # --- Dissolution rate (Noyes-Whitney + Hixson-Crowell) ---
        concentration_gradient = max(0, self.C_s - (M_GI / self.V_GI))
        dissolution_rate = self.k_d * (M_solid ** (2 / 3)) * concentration_gradient

        # Ensure dissolution stops when solid is gone
        if M_solid <= 0:
            dissolution_rate = 0

        # --- Absorption and Elimination rates ---
        absorption_rate = self.k_a * M_GI
        elimination_rate = self.k_e * M_blood

        # Mass balance differential equations
        dM_solid_dt = -dissolution_rate
        dM_GI_dt = dissolution_rate - absorption_rate
        dM_blood_dt = absorption_rate - elimination_rate

        return [dM_solid_dt, dM_GI_dt, dM_blood_dt]

    def simulate(self, t_span, num_points=500):
        """
        Solve the ODE system over the specified time span
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        y0 = [self.dose, 0.0, 0.0]  # Initial state: All drug is solid

        solution = solve_ivp(
            fun=self.system_dynamics,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='Radau'
        )

        # --- Data frame conversion
        results = pd.DataFrame({
            'Time_hr': solution.t,
            'Solid_Mass_mg': solution.y[0],
            'GI_Mass_mg': solution.y[1],
            'Blood_Mass_mg': solution.y[2],
            'Blood_Conc_mg_L': solution.y[2] / self.V_d
        })

        return results

# --- Execution ---

model = MechanisticPKModel(
    dose=500,     # mg (Initial tablet dose)
    k_d=0.5,      # 1/hr (Dissolution rate)
    C_s=100,      # mg/L (Solubility limit)
    V_GI=1.5,     # L (GI fluid volume)
    k_a=0.8,      # 1/hr (Absorption rate into blood)
    k_e=0.15,     # 1/hr (Elimination rate - approx 4.6 hr half-life)
    V_d=50        # L (Volume of distribution)
)

# --- Running simulation for 24 hr period ---
results = model.simulate(t_span=(0, 24))

# --- Plotting mass transfer through the compartments ---
plt.figure(figsize=(10, 6))
plt.plot(results['Time_hr'], results['Solid_Mass_mg'], label='Solid Drug (Tablet)', linestyle='--')
plt.plot(results['Time_hr'], results['GI_Mass_mg'], label='Dissolved in GI Tract')
plt.plot(results['Time_hr'], results['Blood_Mass_mg'], label='Systemic Circulation (Blood)', linewidth=2)

plt.xlabel('Time (Hours)')
plt.ylabel('Mass of API (mg)')
plt.title('Pharmacokinetic Profile of 500mg Solid Oral Dose')
plt.legend()
plt.grid(True)
plt.show()


# --- DOSE OPTIMIZATION ---

def optimize_dosage(target_cmax, bounds=(50, 2000)):
    """
    Finds the ideal tablet dose (mg) to achieve a target maximum
    blood plasma concentration (C_max).
    """

    # --- Objective Function Definition ---
    def objective_function(trial_dose):
        # Trial Dose (Instantiate Model)
        model = MechanisticPKModel(
            dose=trial_dose,
            k_d=0.5,
            C_s=100,
            V_GI=1.5,
            k_a=0.8,
            k_e=0.15,
            V_d=50
        )

        # Run the simulation
        results = model.simulate(t_span=(0, 24))

        # Extract the maximum concentration achieved
        simulated_cmax = results['Blood_Conc_mg_L'].max()

        # Calculate the Mean Squared Error (MSE) penalty
        error = (simulated_cmax - target_cmax) ** 2
        return error

    # --- Run the Scipy Optimizer ---
    print(f"Optimizing dose for Target C_max = {target_cmax} mg/L...")
    result = minimize_scalar(
        objective_function,
        bounds=bounds,
        method='bounded'
    )

    if result.success:
        optimal_dose = result.x
        print(f"Optimization Successful!")
        print(f"Ideal Dosage: {optimal_dose:.2f} mg")
        return optimal_dose
    else:
        print("Optimization failed to converge.")
        return None

# --- Execution ---

# --- Example test for a target concentration of 8.5 mg/L ---
target_concentration = 8.5
ideal_dose = optimize_dosage(target_cmax=target_concentration)

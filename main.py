import time
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from cleanNewProject.config import Modelvariables
from scipy.stats import qmc
from cleanNewProject.likelihood import likelihood
from cleanNewProject.predictor import predictor
from cleanNewProject.minimize import minimize as minimizeCustom
from cleanNewProject.optimize_Own import optimize_expensive
from cleanNewProject.Exact.exact_solution import burgers_viscous_time_exact1
import csv

class StopOptimization(Exception):
    pass

# Callback function to stop optimization
def callback(hyp):
    value, gradient = likelihood(hyp)
    if value is None or gradient is None:
        raise StopOptimization("None value encountered, stopping optimization")

def init_pde_solution_func(x):
    u = np.sin(-np.pi * x)
    return u

dim = Modelvariables['D']
lb = np.array([-1] * dim)  # Lower bounds
ub = np.array([1] * dim)  # Upper bounds

Modelvariables['noise_b'] = 0
Modelvariables['noise_n'] = 0.1

init_guess = np.log([4, 4, np.exp(-6)])

Modelvariables['hyp'] = init_guess

nu = 0.01/np.pi
dt = 1e-3

Modelvariables['nu'] = nu
Modelvariables['dt'] = dt

number_measured_points = 57
Modelvariables['Nmp'] = number_measured_points

number_generated_points = 57
Modelvariables['Ngp'] = number_generated_points

jitter = 1e-7
Modelvariables['jitter'] = jitter

Modelvariables['S0'] = np.zeros((number_measured_points, number_measured_points))

T = 1
nsteps = int(T / Modelvariables['dt'])

num_plots = 8

# Parameters _Exact Solution
vxn = number_generated_points
vx = np.linspace(-1.0, 1.0, vxn)

vt = np.linspace(0.0, 1.0, nsteps)

x_b = np.array([[-1], [1]])
Modelvariables['x_b'] = x_b

u_b = np.array([[0], [0]])
u_b = u_b + Modelvariables['noise_b'] * np.random.randn(*u_b.shape)
Modelvariables['u_b'] = u_b

xstar = np.linspace(-1, 1, number_generated_points).reshape(-1, 1)

sampler = qmc.LatinHypercube(d=dim)
lhs_samples = sampler.random(number_measured_points)
# Scale samples to the range [lb, ub]
x_u = lb + lhs_samples * (ub - lb)
#x_u = xstar
Modelvariables['x_u'] = x_u
init_x_u = x_u

u = init_pde_solution_func(x_u)
u = u + Modelvariables['noise_n'] * np.random.randn(*u.shape)
Modelvariables['u'] = u
init_u = u

# Compute the solution
vu = burgers_viscous_time_exact1(nu, vxn, vx, nsteps, vt)

# Plot the initial data
time_indices = np.linspace(0, nsteps - 1, num_plots, dtype=int)
fig, axes = plt.subplots(num_plots // 2, 2, figsize=(15, 20))
axes = axes.flatten()

j = 0
axes[j].plot(vx, vu[:, 0], label=f'Exact Solution')
axes[j].set_xlabel('Space (x)')
axes[j].set_ylabel('U(x,t)')
axes[j].grid()
axes[j].set_ylim([-1.5, 1.5])
axes[j].plot(init_x_u, init_u, 'rx', linewidth=1, label="measured")
#axes[j].plot(xstar, init_pde_solution_func(xstar), 'b', linewidth=1, label="init Exact Solution")
axes[j].legend()
axes[j].set_title(f"Time: {0}s")
plt.savefig(f'./Figures/Burgers_Final_TrainingsPoints_{number_generated_points}_nu_{nu}.png', dpi=300)

plots_equal_distance = np.linspace(0, nsteps - 1, num_plots, endpoint=True, dtype=int)

# To store the errors
errors = np.zeros(len(vt))

# Initialize timing
start_time = time.time()
method = "CG"
for step, delta_t in enumerate(vt):
    # Run the optimization (commented out as per your instructions)
    try:
        result = minimize(
            lambda hyp: likelihood(hyp)[0],
            Modelvariables['hyp'],
            jac=lambda hyp: likelihood(hyp)[1],
            method=method,
            callback=lambda hyp: callback(hyp),
            options={'maxiter': 5000}
        )

        Modelvariables['hyp'] = result.x
        print("Optimization result:", result)
        print("Optimized hyperparameters:", result.x)

    except StopOptimization as e:
        print("Optimizing failed")
        print(str(e))

    NLML, DNLML = likelihood(Modelvariables['hyp'])  # damit neues K und L generiert wird

    print("Optimal NLML Value:", NLML)
    print("Optimal DNLML Value:", DNLML)

    Kpred, Kvar = predictor(xstar)

    sampler = qmc.LatinHypercube(d=dim)
    lhs_samples = sampler.random(number_generated_points)
    # Scale samples to the range [lb, ub]
    x_u = lb + lhs_samples * (ub - lb)
    #x_u = xstar
    Modelvariables['u'], Modelvariables['S0'] = predictor(x_u)
    Modelvariables['x_u'] = x_u

    if step in plots_equal_distance:
        if step != 0:
            j += 1
            axes[j].set_title(f"Time: {delta_t:.2f}s")
            axes[j].plot(vx, vu[:, step], label=f'Exact Solution')
            axes[j].set_xlabel('Space (x)')
            axes[j].set_ylabel('U(x,t)')
            axes[j].grid()
            axes[j].set_ylim([-1.5, 1.5])

            std_dev = np.sqrt(np.diag(np.abs(Kvar)))
            axes[j].plot(xstar, Kpred, 'r--', linewidth=1, label="predicted")

            # Plot uncertainty as shaded region
            axes[j].fill_between(xstar.flatten(), (Kpred[:, 0] - 1.96 * std_dev), (Kpred[:, 0] + 1.96 * std_dev), color='red', alpha=0.2)
            axes[j].legend()
            plt.tight_layout()
            plt.savefig(f'./Figures/Burgers_Final_TrainingsPoints_{number_generated_points}_nu_{nu}.png', dpi=300)
            #plt.show()
            print("!!!!!!!!!!!PLOTTET!!!!!!!!!!")

    #calculate the error (Absolute error)
    errors[step] = np.linalg.norm((Kpred[:, 0] - vu[:, step])) /np.linalg.norm(vu[:, step])
    print(f'Step: {step}, Time = {delta_t:.6f}, NLML = {NLML.item():.6e}, error_u = {errors[step].item():.6e}')


    # Calculate and print progress and estimated time remaining
    elapsed_time = time.time() - start_time
    percent_complete = (step + 1) / nsteps
    estimated_time_per_percent_complete = elapsed_time / percent_complete
    percented_remaining = 1 - percent_complete
    estimated_time_remaining = estimated_time_per_percent_complete * percented_remaining
    print(f"Progress: {percent_complete:.2%} complete. Estimated time remaining: {estimated_time_remaining:.2f} seconds.")

plt.tight_layout()
plt.savefig(f'./Figures/Burgers_Final_TrainingsPoints_{number_generated_points}_nu_{nu}_optmethod_{method}.png', dpi=300)
plt.show()



# Plotting the error over time
plt.figure(figsize=(10, 6))
plt.plot(vt, errors, label='Relative Error (Normalized L2 Norm)', color='b', linestyle='-', marker='o')
plt.xlabel('Time')
plt.ylabel('Relative Error')
plt.title('Error Between Predicted and Exact Solutions Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f'./Figures/Burgers_Relative_Error_Over_Time.png', dpi=300)
plt.show()

# Save the final error values
with open('./Figures/Burgers_Final_TrainingsPoints_Errors.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Step', 'Absolut_Error'])
    writer.writerows(enumerate(errors))


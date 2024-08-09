import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import GaussianProcess as GP
import csv
import os

# Directory to store the results
results_dir = "Results_linPendel"
os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

# CSV file to store the results
csv_file = os.path.join(results_dir, "results_continuous_omega_sqaured_equal_10.csv")

# Input variables
amplitude = 1
omega_squared = 10
omega = np.sqrt(omega_squared)
T = 2 * np.pi / omega
phase = np.pi / 2
num_observed_values = 5
initial_guesses = [np.sqrt(10), np.sqrt(10.1), np.sqrt(11.5), np.sqrt(9.7), np.sqrt(8.8)]
#initial_guesses = np.arange(0.001,25,0.1)
#initial_guesses = np.sqrt(initial_guesses)
bounds = [(np.sqrt(0), np.sqrt(30))]  # Set bounds for the hyperparameters

# Function to append data to the CSV file
def append_to_csv(data, headers, csv_file):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)  # Write headers if file does not exist
        writer.writerow(data)

# Kernel function definition
def kernel_function(s, t, hyperparameters, variant=1):
    omega = hyperparameters[0]
    length_scale = hyperparameters[0]

    if variant == 1:
        kernel = (np.sin(omega * abs(t - s))) + (np.cos(omega * abs(t - s)))
    else:
        kernel = np.exp(-0.5 * (s - t) ** 2 / length_scale ** 2)

    return kernel

# Linear pendulum solution
def linear_pendulum_solution(t, omega, phase, amplitude):
    return amplitude * np.sin(omega * t + phase)

# Mean function returns zeros
def mean_function(t):
    return np.zeros(len(t))

# Function to minimize for optimal hyperparameters
def minimize_hyperparameters(minimizing_function, initial_guess, bounds, *args, method="L-BFGS-B"):
    tried_values = []

    def callback(xk):
        tried_values.append((xk, minimizing_function(xk, *args)))

    result = minimize(minimizing_function, initial_guess, args=args, method=method, bounds=bounds, callback=callback)
    opt_hyperparameter = result.x
    opt_func_value = result.fun
    return opt_hyperparameter, opt_func_value, tried_values

# Compute the prior covariance matrix
def prior_covariance_matrix(kernel_function, x1, x2, hyperparameters, variant=1) -> np.array:
    return np.array([[kernel_function(a, b, hyperparameters, variant) for a in x2] for b in x1])

# Compute the negative log marginal likelihood
def negative_log_marginal_likelihood(hyperparameters, *args, variant=1):
    y = args[0]
    s = args[1]
    t = args[2]

    n = len(y)
    normalizing_factor = n / 2 * np.log(2 * np.pi)
    covariance_matrix = prior_covariance_matrix(kernel_function, s, t, hyperparameters, variant)

    if variant == 1:
        # Add a small jitter to the diagonal for numerical stability
        jitter = 1e-5
        covariance_matrix += np.eye(len(covariance_matrix)) * jitter

    det_cov_mat = abs(np.linalg.det(covariance_matrix))

    if det_cov_mat <= 0:
        return np.inf  # Penalize this hyperparameter set

    penalty_term = np.log(det_cov_mat) / 2
    probability_term = np.dot(y.T, np.linalg.solve(covariance_matrix, y)) / 2

    log_p = normalizing_factor + penalty_term + probability_term
    return log_p

# Function to fit and plot GPR for both kernel variants
def fit_and_plot_GPR(initial_guess, variant, observed_times, observed_amplitudes, t, phi):
    # Set up GPR
    gpr = GP.GaussianProcessRegression(mean_function, lambda s, t, h: kernel_function(s, t, h, variant), [initial_guess], 0.00001)
    gpr.fit(observed_times, observed_amplitudes)

    # Get mean and covariance from GPR
    mean, covariance_matrix = gpr.predict(t)
    std = np.sqrt(np.abs(np.diag(covariance_matrix)))

    # Optimization to find optimal hyperparameters
    opt_hyperparameters, func_value, tried_values = minimize_hyperparameters(
        lambda h, *args: negative_log_marginal_likelihood(h, *args, variant), [initial_guess], bounds, observed_amplitudes, observed_times, observed_times,
        method="L-BFGS-B"
    )
    print(f"Kernel Variant {variant} with Initial Guess {initial_guess} - Optimal hyperparameters:", opt_hyperparameters)
    print(f"Kernel Variant {variant} with Initial Guess {initial_guess} - Optimal function value:", func_value)

    # Plot actual solution and GPR prediction
    plt.figure(figsize=(10, 6))
    plt.plot(t, phi, label="Exact Solution")
    plt.plot(t, mean, label="Predicted Values")
    plt.scatter(observed_times, observed_amplitudes, label="Observed Values")
    plt.fill_between(t, mean - 1.96 * std, mean + 1.96 * std, alpha=0.5, label="Uncertainty")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    rounded_initial = round(initial_guess, 2)
    rounded_optimal = round(opt_hyperparameters[0], 2)
    plt.title(f'Predicted vs Actual Values\n(Kernel Variant {variant}, Initial Guess {rounded_initial}, Optimized {rounded_optimal})')
    plt.legend()
    plt.ylim([-3, 3])
    plt.savefig(os.path.join(results_dir, f'predicted_vs_actual_variant_{variant}_initial_{rounded_initial}.png'))
    plt.show()
    plt.close()

    # Plot tried values during optimization
    tried_x = [x[0][0] for x in tried_values]
    tried_y = [x[1] for x in tried_values]

    plt.figure(figsize=(10, 6))
    plt.plot(tried_x, tried_y, 'o-', label="Tried Values During Optimization")
    plt.xlabel("Hyperparameter")
    plt.ylabel("Negative Log Marginal Likelihood")
    plt.title(f'Tried Values During Optimization\n(Kernel Variant {variant}, Initial Guess {rounded_initial})')
    plt.scatter(rounded_optimal, tried_values[-1][1], color='red', label="Optimal Value")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'tried_values_optimization_variant_{variant}_initial_{rounded_initial}.png'))
    plt.show()
    plt.close()

    # Set up GPR with optimized hyperparameters
    optimized_gpr = GP.GaussianProcessRegression(mean_function, lambda s, t, h: kernel_function(s, t, h, variant), [opt_hyperparameters[0]], 0.00001)
    optimized_gpr.fit(observed_times, observed_amplitudes)

    # Get mean and covariance from GPR
    optimized_mean, optimized_covariance_matrix = optimized_gpr.predict(t)
    optimized_std = np.sqrt(np.abs(np.diag(optimized_covariance_matrix)))

    # Plot optimized predictions
    plt.figure(figsize=(10, 6))
    plt.plot(t, phi, label="Exact Solution")
    plt.plot(t, optimized_mean, label="Optimized Predicted Values")
    plt.scatter(observed_times, observed_amplitudes, label="Observed Values")
    plt.fill_between(t, optimized_mean - 1.96 * optimized_std, optimized_mean + 1.96 * optimized_std, alpha=0.5, label="Uncertainty")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Optimized Predictions vs Actual Values\n(Kernel Variant {variant}, Initial Guess {rounded_initial}, Optimized {rounded_optimal})')
    plt.legend()
    plt.ylim([-3, 3])
    plt.savefig(os.path.join(results_dir, f'optimized_predictions_variant_{variant}_initial_{rounded_initial}.png'))
    plt.show()
    plt.close()

    # Save data to CSV
    data = [
        variant,
        initial_guess,
        opt_hyperparameters[0],
        list(observed_times),
        list(observed_amplitudes),
        list(mean),
        list(optimized_mean),
        list(tried_x),
        list(tried_y)
    ]
    headers = ["Variant", "Initial Guess", "Optimized Hyperparameter", "Observed Times", "Observed Amplitudes", "Predicted Values", "Optimized Values", "Tried Hyperparameters", "Tried Values"]
    append_to_csv(data, headers, csv_file)

# Set up variables for linear pendulum solution
t = np.linspace(0, 5 * T, 200)

# Solution of the differential equation
phi = linear_pendulum_solution(t, omega, phase, amplitude)

# Pick observed values of exact solution
observed_times = np.random.choice(t, num_observed_values, replace=False)
observed_amplitudes = linear_pendulum_solution(observed_times, omega, phase, amplitude)

# Fit and plot for both kernel variants and multiple initial guesses
for i, initial_guess in enumerate(initial_guesses):
    print(f"Testing initial guess {i + 1}: {initial_guess}")
    fit_and_plot_GPR(initial_guess, 1, observed_times, observed_amplitudes, t, phi)
    fit_and_plot_GPR(initial_guess, 2, observed_times, observed_amplitudes, t, phi)


import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 1/100  # time step (s)
T = 3.0    # total duration (s)
g = 9.81   # gravity (m/s^2)
n_steps = int(T / dt)

# Initial true state [x, y, z, vx, vy, vz]
x_true = np.array([0.0, 0.0, 0.0, 5.0, 2.0, 10.0])
states_true = []
measurements = []

# Noise parameters
Q = np.diag([0.01]*6)                # process noise (small)
R = np.diag([0.051, 0.051, 0.051])         # measurement noise (moderate)

np.random.seed(42)  # reproducibility

# Simulate true trajectory and noisy measurements
for _ in range(n_steps):
    x_true[0:3] += x_true[3:6] * dt
    x_true[5] -= g * dt
    x_true[2] -= 0.5 * g * dt**2
    states_true.append(x_true.copy())

    z = x_true[0:3] + np.random.multivariate_normal(np.zeros(3), R)
    measurements.append(z)

states_true = np.array(states_true)
measurements = np.array(measurements)

# EKF Setup
F = np.eye(6)
F[0, 3] = F[1, 4] = F[2, 5] = dt

H = np.zeros((3, 6))
H[0, 0] = H[1, 1] = H[2, 2] = 1

# Run EKF on first 10 measurements
n_initial = 10
x_est = np.array([0.0, 0.0, 0.0, 4.5, 1.5, 9.5])
P = np.eye(6)
estimates_short = []

for i in range(n_initial):
    z = measurements[i]

    # Prediction
    x_est = F @ x_est
    x_est[5] -= g * dt
    x_est[2] -= 0.5 * g * dt**2
    P = F @ P @ F.T + Q

    # Update
    y = z - H @ x_est
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_est = x_est + K @ y
    P = (np.eye(6) - K @ H) @ P

    estimates_short.append(x_est.copy())

# Future prediction for 50 steps
n_future = 200
future_preds = []
x_pred = estimates_short[-1].copy()

for _ in range(n_future):
    x_pred[0:3] += x_pred[3:6] * dt
    x_pred[5] -= g * dt
    x_pred[2] -= 0.5 * g * dt**2
    future_preds.append(x_pred.copy())

future_preds = np.array(future_preds)
estimates_short = np.array(estimates_short)

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states_true[:, 0], states_true[:, 1], states_true[:, 2], label='True Path', color='blue')
ax.plot(measurements[:n_initial, 0], measurements[:n_initial, 1], measurements[:n_initial, 2], '.', label='Initial Measurements', color='gray')
ax.plot(estimates_short[:, 0], estimates_short[:, 1], estimates_short[:, 2], label='EKF Estimate (10 Steps)', color='red')
ax.plot(future_preds[:, 0], future_preds[:, 1], future_preds[:, 2], '--', label='Future Prediction (50 Steps)', color='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D EKF with Limited Measurements and Future Prediction")
ax.legend()
plt.tight_layout()
plt.show()

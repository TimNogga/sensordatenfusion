import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# hier noch ne quelle für sketchy matrizen hoffe F und Q passt so... https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb

dt = 1.0
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

sigma_a = 1.0
Q = sigma_a ** 2 * np.array([
    [0.25 * dt ** 4, 0, 0.5 * dt ** 3, 0],
    [0, 0.25 * dt ** 4, 0, 0.5 * dt ** 3],
    [0.5 * dt ** 3, 0, dt ** 2, 0],
    [0, 0.5 * dt ** 3, 0, dt ** 2]
])

H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

sigma_measure = 200000
R = np.array([
    [sigma_measure , 0],
    [0, sigma_measure]
])

x = np.array([0, 0, 1, 1])
P = np.eye(4) * 500
n_steps = 500

t = np.linspace(0, 1000, 10000)


def g(t, v, q):
    omega = q / (2 * v)
    A = v ** 2 / q
    x = A * np.sin(omega * t)
    y = A * np.sin(2 * omega * t)
    return x, y


v1, q1 = 300, 9
x_true, y_true = g(t, v1, q1)
ground_truth = np.vstack((x_true, y_true)).T


def simulate_measurements(ground_truth, cov):
    noise = np.random.multivariate_normal([0, 0], cov, ground_truth.shape[0])
    measurements = ground_truth + noise
    return measurements


measurements = simulate_measurements(ground_truth, np.diag([sigma_measure, sigma_measure]))[:n_steps]
estimates = np.zeros((n_steps, 4))
P_estimates = np.zeros((n_steps, 4, 4))

for k in range(n_steps):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    z = measurements[k]
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = (np.eye(4) - K @ H) @ P_pred

    estimates[k] = x
    P_estimates[k] = P

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('PDF')
ax.set_xlim(ground_truth[:, 0].min() - 100, ground_truth[:, 0].max() + 100)
ax.set_ylim(ground_truth[:, 1].min() - 100, ground_truth[:, 1].max() + 100)
ax.set_zlim(0, 0.001)

ax.plot(ground_truth[:, 0], ground_truth[:, 1], zs=0, color='blue', label='Ground Truth')
ax.scatter(measurements[:n_steps, 0], measurements[:n_steps, 1], zs=0, color='red', marker='x', label='Measurements')


def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    return np.exp(-fac / 2) / N


def animate(frame):
    ax.cla()
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('PDF')
    ax.set_xlim(ground_truth[:, 0].min() - 100, ground_truth[:, 0].max() + 100)
    ax.set_ylim(ground_truth[:, 1].min() - 100, ground_truth[:, 1].max() + 100)
    ax.set_zlim(0, 0.001)
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], zs=0, color='blue', label='Ground Truth')
    ax.scatter(measurements[:n_steps, 0], measurements[:n_steps, 1], zs=0, color='red', marker='x',
               label='Measurements')
    ax.plot(estimates[:frame + 1, 0], estimates[:frame + 1, 1], zs=0, color='green', label='Kalman Filter Estimate')

    mean = estimates[frame, :2]
    cov = P_estimates[frame, :2, :2]

    x_vals = np.linspace(mean[0] - 2000, mean[0] + 2000, 100)
    y_vals = np.linspace(mean[1] - 2000, mean[1] + 2000, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = multivariate_gaussian(pos, mean, cov) * 40 #bisschen skaliert für bessere lesbarkeit
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6)


ani = FuncAnimation(fig, animate, frames=n_steps, blit=False, interval=1000, repeat=True)
plt.show()

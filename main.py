import numpy as np
from hurst import Hurst 


def simulate_brownian_motion(n_steps: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal(n_steps)  # random walk
    trajectory = np.cumsum(increments)         # brownian trajectory
    return increments, trajectory


def main():
    hello_msg = (
        "Time series simulation and analysis with Hurst exponent by g00day\n"
        + "-" * 65
        + "\nSimulates Brownian motion and estimates its Hurst exponent (R/S analysis)."
    )
    print(hello_msg)

    n_steps = int(input("n steps: "))

    increments, trajectory = simulate_brownian_motion(n_steps)
        
    model = Hurst(increments)
    h, c = model.calculate_hurst_coefficients(min_n=10, step=5)
    model.plot_rs_regression()
    print(f"Estimated H = {h:.4f}, C = {c:.4f}")
    print(f"Generated {n_steps} steps. Final position: {trajectory[-1]:.4f}")


if __name__ == "__main__":
    main()

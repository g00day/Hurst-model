import numpy as np

from matplotlib import pyplot as plt

class Hurst:
    def __init__(self, x, c=None):
        self.x = np.asarray(x)
        self.c = c

        self.h = None # defaultly is None but can be provided by calculate_hurst_coefficients

        self.log_n = None
        self.log_rs = None

    def _calculate_normalized_ranges(self, min_n, step):
        x = np.asarray(self.x)
        total_len = len(x)

        n_values = []
        rs_values = []

        # Iterate through subinterval sizes from min_n up to half of the series length
        for n in range(min_n, total_len // 2 + 1, step):
            num_blocks = total_len // n
            if num_blocks == 0:
                break

            # Truncate the tail of the series to divide it into equal blocks of size n
            # and reshape into a matrix (num_blocks x n)
            truncated_x = x[: num_blocks * n]
            blocks = truncated_x.reshape(num_blocks, n)

            # vectorized calculation for all blocks
            means = np.mean(blocks, axis=1, keepdims=True)
            cum_deviations = np.cumsum(blocks - means, axis=1)

            r_n = np.ptp(cum_deviations, axis=1)
            s_n = np.std(blocks, axis=1, ddof=1)

            # Exclude blocks with zero standard deviation
            valid_mask = s_n > 0
            if np.any(valid_mask):
                rs_block = r_n[valid_mask] / s_n[valid_mask]
                n_values.append(n)
                rs_values.append(np.mean(rs_block))

        return n_values, rs_values


    def _compute_regression_coefficients(self, min_n=2, step=1):
        """ According to emperical Hurst rule. The normalized ranges as:  
        (R/S)_n = C * n ^ H  
        -----------------------------------------------------------------  
        Basically it is a simple linear dependence of n  
        that can be approximated to linear regression  
        and it's coefficients(H, C) can be found by least squares method  
        H - k
        C - b
        -----------------------------------------------------------------
        Log-transformed linear form: log(R/S)_n = H * log(n) + log(C)
        """
        min_n = max(2, int(min_n))
        step = max(1, int(step))

        n_values, rs_values = self._calculate_normalized_ranges(min_n, step)

        if len(n_values) < 2:
            raise ValueError("not enough serie lenght to build approximation")

        # transform into logarithmic coordinates
        log_n = np.log(n_values)
        log_rs = np.log(rs_values)

        h, log_c = np.polyfit(log_n, log_rs, 1)
        c = np.exp(log_c)

        # save coeffiecients
        self.h = h
        self.c = c

        return h, c


    def calculate_hurst_coefficients(self, min_n=2, step=1):
        c = self.c

        # check if the c was passed while initialization
        if c is not None:

            n_values, rs_values = self._calculate_normalized_ranges(min_n, step)
            n_log, rs_log = np.log(n_values), np.log(rs_values)

            self.log_n = n_log
            self.log_rs = rs_log

            shifted_rs_log = self.log_rs - np.log(self.c)

            # least squares method without bias
            h = np.sum(n_log * shifted_rs_log) / np.sum(n_log ** 2)
            self.h = h

            return h
        
        return self._compute_regression_coefficients(min_n, step)


    def plot_rs_regression(self, min_n=2, step=1):
        if self.h is None or self.log_n is None:
            self.calculate_hurst_coefficients(min_n=min_n, step=step)

        # linear regression log(R/S) = H * log(n) + log(C)
        fit_line = self.h * self.log_n + np.log(self.c)

        # pivot line of random brownian walk(H = 0.5) that goes through the cloud center
        ref_intercept = np.mean(self.log_rs) - 0.5 * np.mean(self.log_n)
        random_walk_line = 0.5 * self.log_n + ref_intercept

        plt.figure(figsize=(9, 6))
        plt.scatter(self.log_n, self.log_rs, color="#1f77b4", alpha=0.75, label="log(R/S)")
        plt.plot(
            self.log_n,
            fit_line,
            color="#d62728",
            linewidth=2.2,
            label=f"regression: H = {self.h:.4f}, C = {self.c:.4f}",
        )
        plt.plot(
            self.log_n,
            random_walk_line,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.5,
            label="Random walk. H = 0.5(Brownian motion)",
        )

        plt.title("R/S graph")
        plt.xlabel("log(n) — log(size of the interval)")
        plt.ylabel("log(R/S) — log(normalized range)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

import numpy as np
import random
import sdm

def gen_parameters():
    return {
        "vOmega": np.random.rand(iP),
        "vA": np.random.rand(iP),
        "vB": np.random.rand(iP),
        "vBeta": np.random.rand(1),
        "dSigma2": np.random.rand(1)
    }

# initialize
iB = 3

# instantiate experiment
iT = 1000
iN = 5
iP = 2

monte_carlo = sdm.MonteCarlo(gen_parameters())

for b in range(iB):
    # store result
    param_est = gen_parameters()
    mRho_sim = np.random.rand(iT,iP)
    mRho_est = np.random.rand(iT,iP)
    convergence = bool(random.getrandbits(1))
    elapsed_time = random.uniform(3, 6)
    monte_carlo.add(param_est, mRho_sim, mRho_est, convergence, elapsed_time)

print(monte_carlo.time_elapsed)
print(monte_carlo.size)
print(monte_carlo.params_dgp)
print(monte_carlo.params_est)
monte_carlo.series_plot
monte_carlo.density_plot

def test_size():
    """test bootstrap sample property"""
    assert monte_carlo.size == iB
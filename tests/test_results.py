import numpy as np
import pandas as pd
import sdm

# initialize input
iT = 1000
iN = 5
iP = 2
model = "dynamic_heterogenous"

params_observation = {
    "vBeta": np.array([1.5]),
    "dSigma2": np.array([2.0])
}
params_filter = {
    "vOmega": np.array([0.1, 0.08]),
    "vA": np.array([0.1, 0.05]),
    "vB": np.array([0.8, 0.6])
}

# create object
monte_carlo = sdm.MonteCarlo(params_observation, params_filter, model)

# test experiment
param_est = {**params_filter, **params_observation}
mRho_sim = np.random.rand(iT+1,iP)
mRho_est = np.random.rand(iT+1,iP)
convergence = True
elapsed_time = 3.16
monte_carlo.add(param_est, mRho_sim, mRho_est, convergence, elapsed_time)

# pandas index
idx = pd.MultiIndex.from_tuples(
    [
        ("vOmega", 0),
        ("vOmega", 1),
        ("vA", 0),
        ("vA", 1),
        ("vB", 0),
        ("vB", 1),
        ("vBeta", 0),
        ("dSigma2", 0)
    ]
)

def test_size():
    """test bootstrap sample property"""
    assert monte_carlo.size == 1

def test_convergence():
    """test convergence property"""
    assert monte_carlo.convergence == [True]

def test_params_est():
    """test estimated parameters property"""
    df_expected = pd.DataFrame.from_dict({0: [0.1, 0.08, 0.1, 0.05, 0.8, 0.6, 1.5, 2.0]}, columns=idx, orient="index")
    df_result = monte_carlo.params_est

    pd.testing.assert_frame_equal(df_expected, df_result)

def test_params_dgp():
    """test estimated parameters property"""
    dict_expected = {**params_filter, **params_observation}
    dict_result = monte_carlo.params_dgp

    assert dict_expected == dict_result
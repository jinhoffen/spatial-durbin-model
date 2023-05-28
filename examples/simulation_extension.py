import numpy as np
import sdm

# define adjacency matrices
matrices = [
    np.array([
        [0,1,1,0,1,1,0,0,0],
        [1,0,1,0,1,1,1,0,0],
        [1,1,0,1,1,0,0,0,1],
        [0,0,1,0,0,0,1,0,0],
        [1,1,1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0],
        [0,1,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1],
        [0,0,1,0,0,0,0,1,0]
    ]),
    np.array([
        [0,1,1,0,1,0,0,0,0],
        [1,0,1,0,1,1,1,0,0],
        [1,0,0,1,1,0,0,0,0],
        [0,1,0,0,0,0,1,0,1],
        [1,1,1,0,0,0,0,0,0],
        [1,1,0,0,0,0,1,0,0],
        [0,1,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1],
        [0,0,1,0,0,1,0,1,0]
    ])
]

# define model parameters
params = {
    "vOmega": np.array([0.05, 0.02]),
    "vA": np.array([0.05, 0.08]),
    "vB": np.array([0.8, 0.75]),
    "vBeta": np.array([1.5]),
    "dSigma2": np.array([2.0])
}

# define model size
iT = 500
iN, _ = matrices[0].shape
iK = len(params["vBeta"])

# define configurations for the spatial gas model
model = sgm.SpatialGasModel("mvnorm", "identity", 0.99)

# simulate covariates
mmX = sgm.covariates.simulate(iT, iN, iK)

# add adjacency matrices and build weighting matrix
adjacency_matrix = sgm.AdjacencyMatrix()
adjacency_matrix.add(0, iT, 0, matrices[0])
adjacency_matrix.add(0, iT, 1, matrices[1])
mmW = adjacency_matrix.build()

# simulate model
simulation = model.simulate(mmX, mmW, params)
simulation_result = simulation.simulate("autoregressive")
mY = simulation_result.mY

# plot simulated time series
sgm.plot_result(mY, simulation_result.mRho, simulation_result.mf)

# exit()

# estimate the model
model_fit = model.estimate(mY, mmX, mmW)

print(model_fit.convergence)
params_est = model_fit.params
print(params_est)

# plot estimated time series
sgm.plot_result(mY, model_fit.mRho, model_fit.mf)

# compare results
params_comparison = sgm.compare(("dgp", params), ("initial", model_fit.params_init), ("estimated", model_fit.params))

print(params_comparison)
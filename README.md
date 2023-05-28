# Spatial Durbin Model with Time Dynamics and Heterogeneity for Panel Data

This package provides a pure Python[^1] implementation of the [Spatial Durbin Model](https://doi.org/10.1111/j.2517-6161.1960.tb00361.x) (SDM) that allows for time-varying and heterogeneous spatial dependence parameters.

This setup is related to Blasques, Koopman, Lucas, Schaumburg (2016) and Zhang, Opschoor, Lucas (2022).

**Table of Contents**

  - [Introduction](#introduction)
  - [Usage](#usage)
    - [Installation](#installation)
    - [Documentation](#documentation)
  - [Citation](#citation)
  - [Contributions](#contributions)
  - [Selected Literature](#selected-literature)
    - [Spatial Durbin Model](#spatial-durbin-model)
    - [Time-varying dependence parameters in Spatial Models](#time-varying-dependence-parameters-in-spatial-models)
    - [Generalized Autoregressive Score (GAS) Models](#generalized-autoregressive-score-gas-models)

## Introduction

The classic SDM is given by

$$y_t = \rho W y_t + X_t \beta + \epsilon_t$$

This package models balanced panel data $y = \lbrace y_{ijt} | i=1,\ldots,I, j=1,\ldots,J, t=1,\ldots,T \rbrace$ where every cross-section has $N = IJ$ units.

The extended model is given by

$$y_t = R_t W_t y_t + X_t \beta + \epsilon_t, \quad \epsilon_t \overset{iid}{\sim} p(\epsilon_t; \Sigma_t, \nu)$$

We define $R_t = \rho_t \otimes \mathbf{I}_ J$ with $\rho_t$ an $I$ dimensional vector.
This assumes group $i$ specific time-varying spatial dependence parameters $\rho_{it}$ for $i=1, \ldots, I$ and $t = 1, \ldots, T$.

We endow the spatial dependence parameter with time series dynamics using the score framework.
This introduces an updating equation for an $I$ dimensional time-varying parameter

$$f_{t+1} = \omega + A s_t + B f_t$$

where $s_t$ is the scaled score of the likelihood function (e.g., see Creal, Koopman, Lucas (2013)).
The link to the spatial dependence parameter is introduced through $\rho_{it} = h(f_{it}) = \gamma \tanh(f_{it})$ with regulation parameter $\gamma \in (0,1)$.

## Usage

### Installation

The package is currently not published to PyPi.
Yet, it can be installed through

```
pip install -e <path-to-root>
```

### Documentation

The documentation is currently not hosted.
It can be compiled with `sphinx` as follows

```
cd <path-to-doc>
make html
```

## Citation

**You may use this code at your own risk when citing this repository.**
You may us the "Cite this repository" link of GitHub.

## Contributions

This project actively accepts contributions.

## Selected Literature

### Spatial Durbin Model

* Durbin, J. (1960), Estimation of parameters in time‐series regression models. Journal of the royal statistical society: Series B (Methodological), 22(1), 139-153. [Source](https://doi.org/10.1111/j.2517-6161.1960.tb00361.x).

### Time-varying dependence parameters in Spatial Models

* Blasques, F., Koopman, S. J., Lucas, A., Schaumburg, J. (2016), Spillover dynamics for systemic risk measurement using spatial financial time series models. Journal of Econometrics, 195(2), 211-223. [Source](https://doi.org/10.1016/j.jeconom.2016.09.001).
* Zhang, X., Opschoor, A., Lucas, A. (2022), The Importance of Heterogeneity in Dynamic Network Models Applied to European Systemic Risk. [Source](https://opschooranne.files.wordpress.com/2022/02/heterogeneity_in_dynamic_networks_2022-1.pdf).

### Generalized Autoregressive Score (GAS) Models

* Creal, D., Koopman, S.J. and Lucas, A. (2013), Generalized Autoregressive Score Models With Applications. Journal of Applied Econometrics, 28: 777-795. [Source](https://onlinelibrary.wiley.com/doi/10.1002/jae.1279).
* Harvey, Andrew C. Dynamic models for volatility and heavy tails: with applications to financial and economic time series. Vol. 52. Cambridge University Press, 2013. [Source](https://www.cambridge.org/core/books/dynamic-models-for-volatility-and-heavy-tails/896F9D5220C4DD2CA675846F888F0BF0).
* Further references: [www.gasmodel.com](http://www.gasmodel.com) (http-only)

[^1]: This code is not optimized for performance!
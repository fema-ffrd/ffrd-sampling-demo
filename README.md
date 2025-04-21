# Stochastic Storm Transposition (SST) Importance Sampling Demos
## Overview
This repository provides a suite of unit tests and simulation routines to explore and compare stochastic storm transposition (SST) methods using importance sampling. The toy problems illustrate how statistical efficiency can be significantly improved over traditional uniform sampling approaches by focusing computational effort on regions of interest—namely, areas where storm footprints intersect a watershed.

![image](https://github.com/user-attachments/assets/8aadf28f-41ba-46d3-8ed5-f0a6a7738882)

## Key Features
- Simple toy problems are provided to demonstrate the sampling techniques.
- Monte Carlo and importance sampling integrations for estimating storm-watershed overlap.
- Adaptive importance sampling based on moment-matching (Cappé et al., 2008).
- Mixture modeling with learned mixing weights (Bugallo et al., 2017).
- Parametric and non-parametric simulations of storm depth and basin-averaged rainfall.
- Depth-frequency curve generation using weighted samples.

## Authors
**Haden Smith**\
USACE Risk Management Center\
[cole.h.smith@usace.army.mil](mailto\:cole.h.smith@usace.army.mil)

---

## Project Structure
The test class `Test_SST_ImportanceSampling` includes the following regions:

### 1. SST Toy Integration Problem
Simple geometric integration of a rectangular watershed using:
- `Test_BasicMonteCarlo_Integration`: baseline MC estimate using uniform sampling.
- `Test_ImportanceSampling_Integration`: basic IS with truncated normal proposals.
- `Test_ImportanceSampling_Integration_ReducedSamples`: IS with fewer samples.
- `Test_Adaptive_ImportanceSampling_Integration_ReducedSamples`: adaptive proposal update via weighted sample moments.

### 2. SST Toy Parametric Simulation Problem
Storms are sampled with varying footprints and rainfall depth:
- `Test_BasicMonteCarlo_Parametric_Simulation`: classic MC with variable storm sizes.
- `Test_ImportanceSampling_Parametric_Simulation_Reduced_Samples`: IS with truncated normal proposals.
- `Test_Adaptive_ImportanceSampling_Parametric_Simulation_ReducedSamples`: adaptive proposal using depth-weighted moment matching and EM-updated mixture weights.

### 3. SST Toy Nonparametric Simulation Problem
Storm depth derived from overlap area and storm intensity:
- `Test_BasicMonteCarlo_Nonparametric_Simulation`: classic MC simulation.
- `Test_ImportanceSampling_Nonparametric_Simulation_Reduced_Samples`: IS using fixed truncated normal proposals.
- `Test_Adaptive_ImportanceSampling_Nonparametric_Simulation_ReducedSamples`: adaptive IS with depth-based weighting and mixture modeling.

---

## Dependencies
This project relies on the following components:
- `Numerics` namespace, which includes:
  - Random number generators (`MersenneTwister`)
  - Distributions (`Uniform`, `TruncatedNormal`, `Normal`, `Mixture`)
  - Statistical tools (`PlottingPositions.Weibull`)

The Numerics library can be downloaded [here](https://github.com/USACE-RMC/Numerics/).

Ensure you have the appropriate numerical and statistical support libraries available in your build environment.

---

## References
- Cappé, O., Douc, R., Guillin, A., Marin, J.M., Robert, C.P. (2008). Adaptive importance sampling in general mixture classes. *The Annals of Statistics*, 36(4), 1947-1976.
- Bugallo, M.F., Elvira, V., Martino, L., Luengo, D., Miguez, J., Djuric, P.M. (2017). Adaptive Importance Sampling: The Past, the Present, and the Future. *IEEE Signal Processing Magazine*, 34(4), 60-79.

---

## Running the Tests
Each method in `Test_SST_ImportanceSampling` is decorated with `[TestMethod]` and can be executed using a compatible test framework (e.g., MSTest).

---

## Usage and Output
Each simulation estimates either an integral or a basin-averaged rainfall quantity and reports:
- Estimated mean value
- Standard error
- A depth-frequency curve (rainfall vs. non-exceedance probability), written to `Debug.WriteLine` as comma-separated values (CSV format)

This CSV output can be visualized using log-scale axes to evaluate simulation accuracy and resolution.

![image](https://github.com/user-attachments/assets/9d320f3f-f025-47d3-bd36-36d07dcdda2e)

---

## Limitations and Future Extensions
- All spatial domains are idealized (square basin and storms), as shown in the image above.   
- These sampling approaches need to be extended to support real-world watershed geometry and rainfall data.
- We still need to explore options for irregularly shaped watersheds and transposition domains. 

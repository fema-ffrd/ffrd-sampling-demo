# FFRD Sampling Demos
## Overview
This repository demonstrates two methodologies for improving computational efficiency in FFRD stochastic storm transposition (SST) simulations:

1. **k-Nearest Neighbors (k-NN) Subsampling**  
Reduces computational burden by selecting representative storm events while preserving multivariate dependence. We stratify in transformed Z-space (Gaussian copula) and subsample within each stratum, then reconstruct frequency curves using the law of total probability.

2. **Importance Sampling (IS)**  
Focuses computational effort on storm realizations that meaningfully intersect the watershed. Synthetic toy problems illustrate how adaptive and mixture-based IS can dramatically improve efficiency over uniform sampling by targeting regions of interest.

## Part 1: k-NN Subsampling

### Concept
The k-NN subsampling approach transforms high-dimensional marginal flows into standardized normal variates (Z-space) via plotting positions. A Gaussian copula captures inter-site dependence in Z-space. We then stratify the joint Z-domain into probability bins, and for each bin, select the K nearest neighbors from a large Monte Carlo ensemble. Finally, we apply the law of total probability to reconstruct marginal and joint frequency curves.

### Implementation

- **Toy Example (`Test_kNN_SubSampling_ToyProblem`)**  
  1. Generate multivariate Normal samples at 5 synthetic sites.  
  2. Convert each marginal to Weibull plotting positions and Z-space.  
  3. Stratify joint Z into 10 bins and subsample K=100 events per bin via k-NN.  
  4. Reconstruct AEP curves using weighted exceedance counts.

- **Case Study (`Test_kNN_SubSampling_Kanawha_14_sites`)**  
  1. Load 25,040 raw SST events across 14 sites.  
  2. Estimate marginal plotting positions and fit a D-dimensional Gaussian copula.  
  3. Stratify joint Z into 10 bins and subsample K=100 events per bin.  
  4. Build site-specific flow frequency curves at pre-defined AEP levels.
 
![image](https://github.com/user-attachments/assets/1f7fef7a-2423-44d5-8b4e-2d57378590cc)

### Key Strengths

- **Multivariate Preservation**: Maintains dependence structure via Gaussian copula in Z-space.  
- **Variance Reduction**: Stratified subsampling reduces estimator variance compared to crude MC at equal sample size.  
- **Computational Savings**: Only K·bins runs needed, instead of N, to approximate frequency curves.

### Limitations

- **Curse of Dimensionality**: In high D, nearest-neighbor neighborhoods become less meaningful without very large N (e.g., ≥100,000).  
- **Bin and K Selection**: Empirical choice of bin count and K trades off bias vs. variance: too few neighbors → high variance; too many → biased smoothing.  
- **Copula Assumption**: Gaussian copula may under-represent tail dependence present in true hydrologic extremes.  
- **Tail Strata Coverage**: Extreme bins may lack sufficient samples, leading to duplicated neighbors or poor representation.

### Potential Improvements

- **Dimensionality Reduction**: Apply PCA or t-SNE to Z-space before k-NN to focus on dominant variance directions [1][2].  
- **Adaptive Binning**: Define strata so each contains roughly equal Monte Carlo counts for balanced variance.  
- **Alternative Sampling**: Use clustering (e.g., k-means) to select representative centroids, or direct multivariate importance sampling to draw from tail-focused copula distributions.  
- **Tail-Copulas**: Replace Gaussian copula with t-copula or extreme-value copulas for better tail dependence modeling [3].

### References

1. Loftsgaarden, D. O., & Quesenberry, C. P. (1965). *A nonparametric estimate of a multivariate density function*. Annals of Mathematical Statistics.  
2. Devroye, L., & Wagner, T. J. (1977). *The L₁ convergence of nearest neighbor density estimates*. Annals of Statistics.  
3. Salvadori, G., De Michele, C., Kottegoda, N. T., & Rosso, R. (2007). *Extremes in hydrology: a review*. Water Resources Research.

---

## Part 2: Importance Sampling

### Concept

Importance Sampling (IS) reroutes sampling effort to high-impact regions (e.g., storms overlapping a watershed). By choosing an appropriate proposal distribution (often a truncated normal or adaptive mixture) IS can dramatically reduce the number of simulations needed for reliable estimates.

### Project Structure
The test class `Test_SST_ImportanceSampling` includes the following regions:

#### 1. SST Toy Integration Problem
Simple geometric integration of a rectangular watershed using:
- `Test_BasicMonteCarlo_Integration`: baseline MC estimate using uniform sampling.
- `Test_ImportanceSampling_Integration`: basic IS with truncated normal proposals.
- `Test_ImportanceSampling_Integration_ReducedSamples`: IS with fewer samples.
- `Test_Adaptive_ImportanceSampling_Integration_ReducedSamples`: adaptive proposal update via weighted sample moments.

#### 2. SST Toy Parametric Simulation Problem
Storms are sampled with varying footprints and rainfall depth:
- `Test_BasicMonteCarlo_Parametric_Simulation`: classic MC with variable storm sizes.
- `Test_ImportanceSampling_Parametric_Simulation_Reduced_Samples`: IS with truncated normal proposals.
- `Test_Adaptive_ImportanceSampling_Parametric_Simulation_ReducedSamples`: adaptive proposal using depth-weighted moment matching and EM-updated mixture weights.

#### 3. SST Toy Nonparametric Simulation Problem
Storm depth derived from overlap area and storm intensity:
- `Test_BasicMonteCarlo_Nonparametric_Simulation`: classic MC simulation.
- `Test_ImportanceSampling_Nonparametric_Simulation_Reduced_Samples`: IS using fixed truncated normal proposals.
- `Test_Adaptive_ImportanceSampling_Nonparametric_Simulation_ReducedSamples`: adaptive IS with depth-based weighting and mixture modeling.

### Key Features

- **Simple Toy Problems**: Simple examples are provided to demonstrate the sampling techniques.
- **Adaptive Moment‐Matching**: Proposal updates via weighted sample moments [4].  
- **Mixture Proposals**: EM‐based mixture weights for flexible tail coverage [5].  
- **Truncated Normal Sampling**: Efficiently targets the watershed‐intersection region.  
- **Frequency Curve Generation**: Computes depth‐AEP curves via weighted exceedance.

![image](https://github.com/user-attachments/assets/8aadf28f-41ba-46d3-8ed5-f0a6a7738882)

### Limitations and Future Extensions
- All spatial domains are idealized (square basin and storms), as shown in the image above.   
- These sampling approaches need to be extended to support real-world watershed geometry and rainfall data.
- Explore options for irregularly shaped watersheds and transposition domains. 

### References

4. Cappé, O., Douc, R., Guillin, A., Marin, J.-M., & Robert, C. P. (2008). *Adaptive importance sampling in general mixture classes*. The Annals of Statistics, 36(4), 1947–1976.  
5. Bugallo, M. F., Elvira, V., Martino, L., Luengo, D., Míguez, J., & Djuric, P. M. (2017). *Adaptive Importance Sampling: The Past, the Present, and the Future*. IEEE Signal Processing Magazine, 34(4), 60–79.

---

## Dependencies

- **Numerics** library suite:  
  - `Numerics.Data.Statistics`  
  - `Numerics.Distributions`  
  - `Numerics.MachineLearning`  
  - `Numerics.Sampling`  
  - `Numerics.Data`  

Provides utilities for multivariate distributions, stratified sampling, k-NN algorithms, and probability transforms.  
[GitHub: USACE-RMC/Numerics](https://github.com/USACE-RMC/Numerics/)

---

## Running the Tests

Use MSTest (or compatible framework) to execute all `[TestMethod]` routines. Results are printed via `Debug.WriteLine` in CSV format (`AEP,Value`) for easy plotting.

---

## Usage and Output

- **Metrics Reported**: Estimated mean, standard error, and depth-frequency (or flow-frequency) curves.  
- **Visualization**: Plot results on log-probability axes to assess tail performance.

![image](https://github.com/user-attachments/assets/9d320f3f-f025-47d3-bd36-36d07dcdda2e)
---

## Authors
**Haden Smith**\
USACE Risk Management Center\
[cole.h.smith@usace.army.mil](mailto\:cole.h.smith@usace.army.mil)

---




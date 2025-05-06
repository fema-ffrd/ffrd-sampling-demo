using Numerics.Data.Statistics;
using Numerics.Distributions;
using Numerics.MachineLearning;
using Numerics.Sampling;
using Numerics;
using System.Diagnostics;
using Numerics.Data;

namespace Test_FFRD
{

    /// <summary>
    /// Demonstrates and evaluates k-nearest neighbors (k-NN) sub-sampling approaches for stochastic storm transposition (SST) simulations.
    /// Includes tests on a toy synthetic dataset and a real-world case study from the Kanawha River Basin.
    /// </summary>
    /// <remarks>
    /// <para>
    ///     <b> Authors: </b>
    ///     Haden Smith, USACE Risk Management Center, cole.h.smith@usace.army.mil
    /// </para>
    /// <para>
    /// This class supports demonstration and performance validation of k-NN sub-sampling techniques
    /// for use in probabilistic frequency analysis and flood hazard modeling. The k-NN routine works in transformed
    /// Z-space using a high-dimensional Gaussian copula.
    /// </para>
    /// </remarks>
    [TestClass]
    public class Test_SST_SubSampling
    {
        /// <summary>
        /// Demonstrates k-NN sub-sampling on a synthetic dataset using Gaussian copula dependence across five flow sites.
        /// Shows how to construct joint probability in Z-space, apply stratified importance sampling, and reconstruct frequency curves.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method generates multivariate Normal samples, transforms to physical flows with specified marginals,
        /// estimates joint probability through a Gaussian copula, and performs stratified k-NN sub-sampling on the joint Z-space.
        /// The result is a probabilistic frequency curve via the law of total probability.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_kNN_SubSampling_ToyProblem()
        {
            // Marginal distributions that represent target flow sites
            var dists = new Normal[] { new Normal(10, 2), new Normal(30, 15), new Normal(17, 5), new Normal(99, 14), new Normal(68, 7) };

            // D = number of target sites; N = number of Monte Carlo events
            int D = 5;
            int N = 100000;

            // Average inter-site correlation 
            // This method works best when correlations are relatively large > 0.5. 
            // You can change rho and compare results. 
            double rho = 0.5; 
            var mean = new double[] { 0, 0, 0, 0, 0 };
            var covar = new double[,]
                            {{ 1, rho, rho, rho, rho },
                             { rho, 1, rho, rho, rho },
                             { rho, rho, 1, rho, rho },
                             { rho, rho, rho, 1, rho },
                             { rho, rho, rho, rho, 1 }};
            var mvn = new MultivariateNormal(mean, covar);
            var rnd = mvn.GenerateRandomValues(N, 12345);

            // Prepare arrays for k-NN features
            var yVals = new double[N];      // Dummy response vector (sum of flows)
            var xVals = new double[N, D];   // Raw flows at each site
            var xPP = new double[N, D + 1]; // Weibull plotting positions + joint prob
            var xZ = new double[N, D + 1];  // Standard-normal transformed positions

            // 2) Populate xVals and yVals
            for (int i = 0; i < N; i++)
            {
                double sum = 0;
                for (int j = 0; j < D; j++)
                {
                    // Sample flow at site j
                    xVals[i, j] = dists[j].InverseCDF(Normal.StandardCDF(rnd[i, j]));
                    sum += xVals[i, j];
                }
                yVals[i] = sum; // Use sum as dummy y for k-NN
            }

            // 3) Convert each site marginal to plotting positions (Weibull) and standard-normal
            for (int j = 0; j < D; j++)
            {
                // Rank data in-place: lowest to highest
                var ranks = Statistics.RanksInPlace(xVals.GetColumn(j));
                // Weibull plotting position: rank / (N+1)
                xPP.SetColumn(j, ranks.Divide(N + 1));
                // Z-transform of plotting positions
                xZ.SetColumn(j, Normal.StandardZ(ranks.Divide(N + 1)));
            }

            // 4) Estimate covariance of site Zs to build Gaussian copula
            var covMatrix = new RunningCovarianceMatrix(D);
            for (int i = 0; i < N; i++)
            {
                // Push each D-dimensional Z-vector (exclude last joint column)
                covMatrix.Push(xZ.GetRow(i).Subset(0, D - 1));
            }
            // Create multivariate normal with estimated mean and scaled covariance
            var mvnZ = new MultivariateNormal(covMatrix.Mean.GetColumn(0), (1d / N * covMatrix.Covariance).ToArray());
            // Create inputs for the product of conditional marginals (PCM) method
            var corr = mvnZ.Covariance;
            var ind = new int[D];
            ind.Fill(1);

            // 5) Compute joint plotting positions via high-dimensional copula CDF
            for (int i = 0; i < N; i++)
            {
                // Joint probability under Gaussian copula
                xPP[i, D] = Probability.JointProbabilityHPCM(xPP.GetRow(i).Subset(0, D - 1), ind, corr);
                // Z-transform of joint probability
                xZ[i, D] = Normal.StandardZ(xPP[i, D]);
            }

            // 6) Build k-NN sampler on (D+1)-dim Z-space, using yVals as dummy
            int K = 100;
            var kNN = new KNearestNeighbors(xZ, yVals, K) { IsRegression = false }; // we only need indices, not predictions

            // 7) Stratify Z-space into bins for the joint Z dimension
            int bins = 10;
            var stratOpts = new StratificationOptions(0.05, 0.9999, bins, true);
            var strata = Stratify.Probabilities(stratOpts, Stratify.ImportanceDistribution.Normal, true);
            var weights = strata.Select(w => w.Weight).ToArray();

            // Prepare storage for sub-sampled flows per site & per bin
            var minMax = new double[2, D];
            var results = new List<double[,]>();
            for (int i = 0; i < D; i++)
            {
                results.Add(new double[bins, K]);
                minMax[0, i] = double.MaxValue;
                minMax[1, i] = double.MinValue;
            }

            // 8) Sub-sample via k-NN within each bin centroid in Z-space
            for (int i = 0; i < bins; i++)
            {
                // Build feature vector for bin "midpoint" in Z-space
                var zQuery = new double[1, D + 1];
                for (int j = 0; j < D; j++)
                {
                    // Marginal Z = StandardZ(bin midpoint)
                    zQuery[0, j] = Normal.StandardZ(strata[i].Midpoint);
                }
                // Joint Z: feed marginals through multivariate CDF then transform
                var jp = Normal.StandardCDF(zQuery.GetRow(0).Subset(0, D - 1));
                zQuery[0, D] = Normal.StandardZ(Probability.JointProbabilityHPCM(jp, ind, corr));

                // Get the K nearest neighbors in Z-space
                var neighborIndices = kNN.GetNeighbors(zQuery);

                // Extract original flows for each site & update min/max
                for (int d = 0; d < D; d++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        double val = xVals[neighborIndices[k], d];
                        results[d][i, k] = val;
                        minMax[0, d] = Math.Min(minMax[0, d], val);
                        minMax[1, d] = Math.Max(minMax[1, d], val);
                    }
                }

            }

            // 9) Post-process each site: reconstruct frequency curves
            var aeps = new double[] { 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005 };
            var curves = new List<double[,]>();

            for (int d = 0; d < D; d++)
            {
                // 9a) Construct an x-grid between observed min & max
                var x = CreateXValues(minMax[0, d], minMax[1, d], 100, true);
                // 9b) Compute AEP curve via Total Probability
                var aepX = CreateFrequencyCurve(x, weights, results[d]);
                // 9c) Interpolate to target AEPs
                curves.Add(InterpolateFrequencyCurve(aeps, aepX, true));

                // Print results for debugging
                for (int i = 0; i < aeps.Length; i++)
                {
                    Debug.WriteLine(aeps[i].ToString() + "," + curves.Last()[i, 1].ToString());
                }

            }

        }

        /// <summary>
        /// Applies the k-NN sub-sampling method to a real-world case study in the Kanawha River Basin using HEC-HMS output. 
        /// </summary>
        /// <remarks>
        /// <para><b>Case Study Summary:</b></para>
        /// <list type="bullet">
        ///     <item>Uses SST-based simulations at 14 sites from 25,040 SST events, where there was an average of 10 events per year, for 2,000 years.</item>
        ///     <item>Applies k-NN sub-sampling in Z-space to reduce the dataset to 1,000 events while preserving multivariate structure.</item>
        ///     <item>Partial duration frequency curves are reconstructed using total probability and stratified sub-samples.
        ///     These results can be converted to annual max using the Langbein conversion where lambda is 10 events per year.</item>
        /// </list>
        /// <para><b>Strengths:</b></para>
        /// <list type="bullet">
        ///     <item>Efficient dimensionality reduction while preserving correlation structure.</item>
        ///     <item>Works with existing peak flow time series data from hydrologic models.</item>
        ///     <item>Generalizable to other outputs like velocity, depth, or damage curves.</item>
        /// </list>
        /// <para><b>Limitations:</b></para>
        /// <list type="bullet">
        ///     <item>Requires moderate to strong hydrologic correlation between sites (ρ > 0.5).</item>
        ///     <item>Bin and neighbor selection currently empirical; adaptive strategies could enhance performance.</item>
        ///     <item>Uses peak flow as the basis for sub-sampling; alternate criteria may be more appropriate in some contexts.</item>
        /// </list>
        /// </remarks>
        [TestMethod]
        public void Test_kNN_SubSampling_Kanawha_14_sites()
        {
            // 1) Load raw CSV into jagged array: columns: [ID, ..., site flows...]
            // Change file location as needed
            var data = ReadCsvFile("C:/Users/Q0RMCCHS/Documents/0_RMC_Work/FFRD/Kanawha Basin Results/simulation_peaks_by_event_primary.csv");

            // D = number of target sites; N = number of Monte Carlo events
            int D = 14;
            int N = data.Length;

            // Prepare arrays for k-NN features
            var yVals = new double[N];      // Dummy response vector (sum of flows)
            var xVals = new double[N, D];   // Raw flows at each site
            var xPP = new double[N, D + 1]; // Weibull plotting positions + joint prob
            var xZ = new double[N, D + 1];  // Standard-normal transformed positions

            // 2) Populate xVals and yVals
            for (int i = 0; i < N; i++)
            {
                double sum = 0;
                for (int j = 0; j < D; j++)
                {
                    // Pick off flow at site j (assuming columns 3..)
                    xVals[i, j] = data[i][j + 3];
                    sum += xVals[i, j];
                }
                yVals[i] = sum;               // Use sum as dummy y for k-NN
            }

            // 3) Convert each site marginal to plotting positions (Weibull) and standard-normal
            for (int j = 0; j < D; j++)
            {
                // Rank data in-place: lowest to highest
                var ranks = Statistics.RanksInPlace(xVals.GetColumn(j));
                // Weibull plotting position: rank / (N+1)
                xPP.SetColumn(j, ranks.Divide(N + 1));
                // Z-transform of plotting positions
                xZ.SetColumn(j, Normal.StandardZ(ranks.Divide(N + 1)));
            }

            // 4) Estimate covariance of site Zs to build Gaussian copula
            var covMatrix = new RunningCovarianceMatrix(D);
            for (int i = 0; i < N; i++)
            {
                // Push each D-dimensional Z-vector (exclude last joint column)
                covMatrix.Push(xZ.GetRow(i).Subset(0, D - 1));
            }
            // Create multivariate normal with estimated mean and scaled covariance
            var mvnZ = new MultivariateNormal(covMatrix.Mean.GetColumn(0), (1d / N * covMatrix.Covariance).ToArray());
            // Create inputs for the product of conditional marginals (PCM) method
            var corr = mvnZ.Covariance;
            var ind = new int[D];
            ind.Fill(1);

            // 5) Compute joint plotting positions via high-dimensional copula CDF
            for (int i = 0; i < N; i++)
            {
                // Joint probability under Gaussian copula
                xPP[i, D] = Probability.JointProbabilityHPCM(xPP.GetRow(i).Subset(0, D - 1), ind, corr);
                // Z-transform of joint probability
                xZ[i, D] = Normal.StandardZ(xPP[i, D]);
            }

            // 6) Build k-NN sampler on (D+1)-dim Z-space, using yVals as dummy
            int K = 100;
            var kNN = new KNearestNeighbors(xZ, yVals, K) { IsRegression = false }; // we only need indices, not predictions

            // 7) Stratify Z-space into bins for the joint Z dimension
            int bins = 10;
            var stratOpts = new StratificationOptions(0.05, 0.9999, bins, true);
            var strata = Stratify.Probabilities(stratOpts, Stratify.ImportanceDistribution.Normal, true);
            var weights = strata.Select(w => w.Weight).ToArray();

            // Prepare storage for sub-sampled flows per site & per bin
            var minMax = new double[2, D];
            var results = new List<double[,]>();
            for (int i = 0; i < D; i++)
            {
                results.Add(new double[bins, K]);
                minMax[0, i] = double.MaxValue;
                minMax[1, i] = double.MinValue;
            }

            // 8) Sub-sample via k-NN within each bin centroid in Z-space
            for (int i = 0; i < bins; i++)
            {
                // Build feature vector for bin "midpoint" in Z-space
                var zQuery = new double[1, D + 1];
                for (int j = 0; j < D; j++)
                {
                    // Marginal Z = StandardZ(bin midpoint)
                    zQuery[0, j] = Normal.StandardZ(strata[i].Midpoint);
                }
                // Joint Z: feed marginals through multivariate CDF then transform
                var jp = Normal.StandardCDF(zQuery.GetRow(0).Subset(0, D - 1));
                zQuery[0, D] = Normal.StandardZ(Probability.JointProbabilityHPCM(jp, ind, corr));

                // Get the K nearest neighbors in Z-space
                var neighborIndices = kNN.GetNeighbors(zQuery);

                // Extract original flows for each site & update min/max
                for (int d = 0; d < D; d++)
                {
                    for (int k = 0; k < K; k++)
                    {
                        double val = xVals[neighborIndices[k], d];
                        results[d][i, k] = val;
                        minMax[0, d] = Math.Min(minMax[0, d], val);
                        minMax[1, d] = Math.Max(minMax[1, d], val);
                    }
                }

            }

            // 9) Post-process each site: reconstruct frequency curves
            var aeps = new double[] { 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005 };
            var curves = new List<double[,]>();

            for (int d = 0; d < D; d++)
            {
                // 9a) Construct an x-grid between observed min & max
                var x = CreateXValues(minMax[0, d], minMax[1, d], 100, true);
                // 9b) Compute AEP curve via Total Probability
                var aepX = CreateFrequencyCurve(x, weights, results[d]);
                // 9c) Interpolate to target AEPs
                curves.Add(InterpolateFrequencyCurve(aeps, aepX, true));

                // Print results for debugging
                for (int i = 0; i < aeps.Length; i++)
                {
                    Debug.WriteLine(aeps[i].ToString() + "," + curves.Last()[i, 1].ToString());
                }

            }

        }

        #region Helpers

        /// <summary>
        /// Reads numeric data from a CSV file into a jagged array of doubles, skipping the header row.
        /// </summary>
        /// <param name="filePath">Path to the CSV file.</param>
        /// <returns>A jagged array representing the numerical values from each row in the CSV file.</returns>
        public double[][] ReadCsvFile(string filePath)
        {
            // Read all lines from the CSV file
            string[] lines = File.ReadAllLines(filePath);

            // Initialize a 2D array to hold the data
            double[][] result = new double[lines.Length - 1][];

            // The first row has headers, so start on second row
            for (int i = 1; i < lines.Length; i++)
            {
                // Split the line by commas (or your chosen delimiter)
                var strings = lines[i].Split(',');
                result[i - 1] = new double[strings.Length];
                for (int j = 0; j < strings.Length; j++)
                {
                    result[i - 1][j] = double.Parse(strings[j]);
                }
            }

            return result;
        }

        /// <summary>
        /// Generates a series of x-values spanning a range for frequency curve plotting.
        /// Can use either linear or logarithmic spacing.
        /// </summary>
        /// <param name="min">Minimum x-value.</param>
        /// <param name="max">Maximum x-value.</param>
        /// <param name="bins">Number of bins or intervals.</param>
        /// <param name="isLogarithmic">If true, values are spaced logarithmically; otherwise, linearly.</param>
        /// <returns>An array of x-values.</returns>

        public double[] CreateXValues(double min, double max, int bins = 100, bool isLogarithmic = false)
        {
            var xBins = Stratify.XValues(new StratificationOptions(min, max, bins, false), isLogarithmic);
            var xList = xBins.Select(b => b.LowerBound).ToList();
            xList.Add(xBins.Last().UpperBound);
            return xList.ToArray();
        }

        /// <summary>
        /// Computes a frequency curve (AEP vs. x) using the law of total probability across sub-sampled strata.
        /// </summary>
        /// <param name="xValues">Grid of x-values at which exceedance probabilities are computed.</param>
        /// <param name="weights">Importance sampling weights for each stratum.</param>
        /// <param name="samples">Sub-sampled values from each stratum [stratum, sample].</param>
        /// <returns>A 2D array where column 0 contains AEP values and column 1 contains x-values.</returns>
        public double[,] CreateFrequencyCurve(double[] xValues, double[] weights, double[,] samples)
        {

            int D = xValues.Length;
            int N = samples.GetLength(0);   // number of strata
            int M = samples.GetLength(1);   // samples per stratum
            var curve = new double[D, 2];   // [AEP, x]

            for (int i = 0; i < D; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    // Count fraction of samples in stratum j exceeding xValues[i]
                    int exceed = 0;
                    for (int k = 0; k < M; k++)
                    {
                        if (samples[j, k] > xValues[i])
                        {
                            exceed++;
                        }
                    }
                    // Accumulate weighted exceedance probability
                    // Update AEP using Total Probability
                    curve[i, 0] += weights[j] * exceed / M;
                }
                curve[i, 1] = xValues[i];
            }

            return curve;
        }

        /// <summary>
        /// Interpolates an empirical frequency curve to match a set of target AEPs.
        /// </summary>
        /// <param name="targetAEPs">Desired AEP values for interpolation.</param>
        /// <param name="curve">Original frequency curve with AEP (col 0) and x (col 1).</param>
        /// <param name="isLogarithmic">If true, interpolate x-values in logarithmic space.</param>
        /// <returns>Interpolated frequency curve aligned with the specified AEPs.</returns>
        public double[,] InterpolateFrequencyCurve(double[] targetAEPs, double[,] curve, bool isLogarithmic = false)
        {
            // Build sorted vectors of (AEP, x)
            var aepList = new List<double> { curve[0, 0] };
            var xList = new List<double> { curve[0, 1] };
            for (int i = 1; i < curve.GetLength(0); i++)
            {
                if (curve[i, 0] != aepList.Last())
                {
                    aepList.Add(curve[i, 0]);
                    xList.Add(curve[i, 1]);
                }
            }

            // Set up linear interpolator in Normal-Z space
            var linInt = new Linear(aepList, xList, SortOrder.Descending) { XTransform = Transform.NormalZ, YTransform = isLogarithmic ? Transform.Logarithmic : Transform.None };

            // Evaluate at target AEPs
            var result = new double[targetAEPs.Length, 2];
            result.SetColumn(0, targetAEPs);
            result.SetColumn(1, linInt.Interpolate(targetAEPs));
            return result;
        }

        #endregion

    }
}

using Numerics;
using Numerics.Data.Statistics;
using Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test_FFRD
{
    /// <summary>
    /// Test jackknife uncertainty approach for SST.
    /// </summary>
    /// <remarks>
    /// <para>
    ///     <b> Authors: </b>
    ///     Haden Smith, USACE Risk Management Center, cole.h.smith@usace.army.mil
    /// </para>
    /// <para>
    /// </para>
    /// </remarks>
    [TestClass]
    public class Test_JackknifeUncertainty
    {

        /// <summary>
        /// Compares the Jackknife confidence intervals to bootstrap confidence intervals.
        /// </summary>
        /// <remarks>
        /// </remarks>
        [TestMethod]
        public void Test_AnnualMaximumSimulation()
        {
            // probabilities to evaluate confidence intervals
            var probabilities = new double[] { 0.9999, 0.9998, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01 };

            var Fs = new GeneralizedExtremeValue(70, 15, -0.05); // Parent Hazard distribution
            int bootReps = 10000; // Number of bootstrap replications
            int N = 40; // typical sample size of AORC-type data sources for SST

            // Create synthetic sample and estimate the parameters
            var sample = Fs.InverseCDF(PlottingPositions.Weibull(N));
            Fs.Estimate(sample, ParameterEstimationMethod.MaximumLikelihood);
            var thetaHat = Fs.InverseCDF(probabilities);
            thetaHat.Apply(Math.Log); // log-transform

            // Run the bootstrap analysis.
            // For accurate 90% CIs, we need to run 10,000 replications
            var bootstrap = new BootstrapAnalysis(Fs, ParameterEstimationMethod.MaximumLikelihood, N, bootReps, 12345);
            var bootstrapCIs = bootstrap.PercentileQuantileCI(probabilities);

            for (int i = 0; i < probabilities.Length; i++)
            {
                Debug.WriteLine(bootstrapCIs[i, 1] + "," + bootstrapCIs[i, 0] + "," + Math.Exp(thetaHat[i]));
            }

            // Run jackknife analysis to get quantile standard errors
            // The jackknife only requires N replications, so it is much faster than the bootstrap. 
            // However, the jackknife will only give us the quantile standard error.
            // We have to approximate the CI using the standard Normal method.
            var logSE = JackkifeStandardError(sample, probabilities, thetaHat, Fs);
            double z95 = Normal.StandardZ(0.95), z05 = Normal.StandardZ(0.05);

            for (int i = 0; i < probabilities.Length; i++)
            {
                // Convert CI back to real-space.
                var ci95 = Math.Exp(thetaHat[i] + z95 * logSE[i]);
                var ci05 = Math.Exp(thetaHat[i] + z05 * logSE[i]);
                Debug.WriteLine(ci95 + "," + ci05 + "," + Math.Exp(thetaHat[i]));
            }

        }


        /// <summary>
        /// Estimates the log-space standard error for each probability using the jackknife method.
        /// </summary>
        /// <param name="sampleData">Sample of data.</param>
        /// <param name="probabilities">List of non-exceedance probabilities.</param>
        /// <param name="thetaHats">The list of best-estimate log-transformed quantiles.</param>
        private double[] JackkifeStandardError(IList<double> sampleData, IList<double> probabilities, IList<double> thetaHats, UnivariateDistributionBase univariateDistribution)
        {
            var N = sampleData.Count;
            var I2 = new double[probabilities.Count];
            var se = new double[probabilities.Count];

            // Perform Jackknife
            Parallel.For(0, N, idx =>
            {
                // Remove data point
                var jackSample = new List<double>(sampleData);
                jackSample.RemoveAt(idx);
                // Estimate distribution
                var newDistribution = univariateDistribution.Clone();

                try
                {
                    ((IEstimation)newDistribution).Estimate(jackSample, ParameterEstimationMethod.MaximumLikelihood);
                    // Get quantiles from new distribution
                    var thetaJack = new double[probabilities.Count];
                    for (int i = 0; i < probabilities.Count; i++)
                    {
                        thetaJack[i] = Tools.Log(newDistribution.InverseCDF(probabilities[i]));
                        Tools.ParallelAdd(ref I2[i], Math.Pow(thetaHats[i] - thetaJack[i], 2));
                    }
                }
                catch
                {
                    // MLE can fail to find a solution
                    // So just skipping as a safeguard for this demo. 
                };

            });
            // Get quantile standard error
            for (int i = 0; i < probabilities.Count; i++)
                se[i] = Math.Sqrt((N - 1) / (double)N * I2[i]);

            return se;
        }


    }
}

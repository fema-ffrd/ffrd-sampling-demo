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
                //Debug.WriteLine(bootstrapCIs[i, 1] + "," + bootstrapCIs[i, 0] + "," + Math.Exp(thetaHat[i]));
            }

            // Run jackknife analysis to get quantile standard errors
            // The jackknife only requires N replications, so it is much faster than the bootstrap. 
            // However, the jackknife will only give us the quantile standard error.
            // We have to approximate the CI using the standard Normal method.
            var logSE = JackknifeStandardError(sample, probabilities, Fs);
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
        private double[] JackknifeStandardError(IList<double> sampleData, IList<double> probabilities, UnivariateDistributionBase univariateDistribution)
        {
            int N = sampleData.Count;
            int P = probabilities.Count;

            // collect jackknife estimates: rows = successful deletions, cols = probabilities
            var thetaJK = new List<double[]>(capacity: N);

            // 1) generate leave-one-out estimates
            Parallel.For(0, N, i =>
            {
                var jack = new List<double>(sampleData);
                jack.RemoveAt(i);

                var d = univariateDistribution.Clone();
                try
                {
                    ((IEstimation)d).Estimate(jack, ParameterEstimationMethod.MaximumLikelihood);

                    var row = new double[P];
                    for (int j = 0; j < P; j++)
                        row[j] = Math.Log(d.InverseCDF(probabilities[j]));

                    lock (thetaJK) thetaJK.Add(row);
                }
                catch
                {
                    // skip this replicate; optionally log/debug
                }
            });

            int m = thetaJK.Count;  // successful replicates
            if (m < 2) throw new InvalidOperationException("Insufficient jackknife replicates.");

            // 2) compute jackknife means
            var mean = new double[P];
            foreach (var row in thetaJK)
                for (int j = 0; j < P; j++) mean[j] += row[j];
            for (int j = 0; j < P; j++) mean[j] /= m;

            // 3) compute sum of squared deviations around jackknife mean
            var se = new double[P];
            for (int j = 0; j < P; j++)
            {
                double ssd = 0.0;
                foreach (var row in thetaJK)
                {
                    double d = row[j] - mean[j];
                    ssd += d * d;
                }
                se[j] = Math.Sqrt((m - 1.0) / m * ssd);
            }
            return se;
        }

    }
}

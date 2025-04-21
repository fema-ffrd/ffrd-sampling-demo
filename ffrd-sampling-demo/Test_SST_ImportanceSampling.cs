using Numerics;
using Numerics.Data.Statistics;
using Numerics.Distributions;
using Numerics.Mathematics.SpecialFunctions;
using Numerics.Sampling;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace Test_FFRD
{

    /// <summary>
    /// A class for testing SST importance sampling methods using toy problems.
    /// </summary>
    /// <remarks>
    /// <para>
    ///     <b> Authors: </b>
    ///     Haden Smith, USACE Risk Management Center, cole.h.smith@usace.army.mil
    /// </para>
    /// <para>
    /// </remarks>
    [TestClass]
    public class Test_SST_ImportanceSampling
    {

        #region SST Toy Integration Problem

        /// <summary>
        /// A simple toy integrand to mimic the logic of stochastic storm transposition (SST).
        /// This represents a rectangular watershed embedded within a larger domain.
        /// The true area of the watershed (integration region) is 9.
        /// </summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        private double SST_Toy_Integrand(double x, double y)
        {
            double result = 0;

            // Watershed bounds: 4 ≤ x < 7 and 3 ≤ y < 6 => Area = 3 * 3 = 9
            if (x < 7 && x >= 4 && y < 6 && y >= 3)
            {
                // Storm falls within the watershed
                result = 1;
            }

            // Otherwise, storm falls outside the watershed and contributes nothing
            return result;
        }

        /// <summary>
        /// Baseline test using crude Monte Carlo integration.
        /// Represents how SST is typically performed today using uniform transpositions.
        /// </summary>
        [TestMethod]
        public void Test_BasicMonteCarlo_Integration()
        {
            // Integration domain: storms are uniformly transposed over [1,21]^2
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin);  // Area = 400

            const int N = 100000;  // Number of random samples
            var prng = new MersenneTwister(12345);  // Random number generator (seeded for reproducibility)
            var xDist = new Uniform(xmin, xmax);
            var yDist = new Uniform(ymin, ymax);

            double sum = 0, sum2 = 0;

            // Perform Monte Carlo integration
            for (int i = 0; i < N; i++)
            {
                // Sample storm location uniformly over the domain
                double x = xDist.InverseCDF(prng.NextDouble());
                double y = yDist.InverseCDF(prng.NextDouble());

                // Evaluate whether the storm intersects the watershed
                double f = SST_Toy_Integrand(x, y);

                // Accumulate results for mean and variance estimation
                sum += f;
                sum2 += f * f;
            }

            double avg = sum / N;
            double avg2 = sum2 / N;

            // Estimate integral by multiplying average value by domain area
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output (N=100,000): ~9.24 ± 0.19
        }

        /// <summary>
        /// Importance Sampling test with 100,000 samples.
        /// We sample more frequently from regions near the watershed to reduce variance.
        /// </summary>
        [TestMethod]
        public void Test_ImportanceSampling_Integration()
        {
            // Integration domain: storms are uniformly transposed over [1,21]^2
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin);  // Area = 400

            const int N = 100000;  // Number of random samples
            var prng = new MersenneTwister(12345);  // Random number generator (seeded for reproducibility)
            var xDist = new Uniform(xmin, xmax);
            var yDist = new Uniform(ymin, ymax);

            // Truncated normal distributions focused on the watershed center
            var xISDist = new TruncatedNormal(5.5, 0.87, xmin, xmax); // Centered in [4,7]
            var yISDist = new TruncatedNormal(4.5, 0.87, ymin, ymax); // Centered in [3,6]

            double sum = 0, sum2 = 0;

            // Perform Monte Carlo integration
            for (int i = 0; i < N; i++)
            {
                // Sample from the importance distribution
                double x = xISDist.InverseCDF(prng.NextDouble());
                double y = yISDist.InverseCDF(prng.NextDouble());

                // Compute importance sampling weight: w = target / proposal
                double p = xDist.PDF(x) * yDist.PDF(y);       // Target: uniform PDF
                double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal: truncated normal PDF
                double w = (q > 0.0) ? p / q : 0.0;

                // Evaluate weighted integrand
                double f = SST_Toy_Integrand(x, y) * w;

                // Accumulate results for mean and variance estimation
                sum += f;
                sum2 += f * f;
            }

            double avg = sum / N;
            double avg2 = sum2 / N;

            // Estimate integral by multiplying average value by domain area
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output: ~8.99 ± 0.025 — significantly lower error than basic Monte Carlo
        }

        /// <summary>
        /// Importance Sampling with reduced sample size (10,000).
        /// Demonstrates how a well-chosen proposal distribution can still perform well with fewer samples.
        /// </summary>
        [TestMethod]
        public void Test_ImportanceSampling_Integration_ReducedSamples()
        {
            // Integration domain: storms are uniformly transposed over [1,21]^2
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin);  // Area = 400

            const int N = 10000;  // Number of random samples is only 10,000
            var prng = new MersenneTwister(12345);  // Random number generator (seeded for reproducibility)
            var xDist = new Uniform(xmin, xmax);
            var yDist = new Uniform(ymin, ymax);

            // Truncated normal distributions focused on the watershed center
            var xISDist = new TruncatedNormal(5.5, 0.87, xmin, xmax); // Centered in [4,7]
            var yISDist = new TruncatedNormal(4.5, 0.87, ymin, ymax); // Centered in [3,6]

            double sum = 0, sum2 = 0;

            // Perform Monte Carlo integration
            for (int i = 0; i < N; i++)
            {
                // Sample from the importance distribution
                double x = xISDist.InverseCDF(prng.NextDouble());
                double y = yISDist.InverseCDF(prng.NextDouble());

                // Compute importance sampling weight: w = target / proposal
                double p = xDist.PDF(x) * yDist.PDF(y);       // Target: uniform PDF
                double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal: truncated normal PDF
                double w = (q > 0.0) ? p / q : 0.0;

                // Evaluate weighted integrand
                double f = SST_Toy_Integrand(x, y) * w;

                // Accumulate results for mean and variance estimation
                sum += f;
                sum2 += f * f;
            }

            double avg = sum / N;
            double avg2 = sum2 / N;

            // Estimate integral by multiplying average value by domain area
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output: ~8.97 ± 0.076 — still significantly lower error than basic Monte Carlo with 100,000 samples

        }

        /// <summary>
        /// Adaptive Importance Sampling with reduced sample size (10,000),
        /// preceded by a 1,000-sample adaptation phase (5 iterations × 200 samples).
        ///
        /// Motivation:
        /// In the prior test, we assumed knowledge of where the proposal distribution should be centered.
        /// In this test, we adapt the proposal distribution iteratively by computing importance weights,
        /// matching moments, and smoothing updates. This mimics realistic SST applications where the
        /// optimal storm transposition region is unknown.
        ///
        /// Key takeaways:
        /// - Efficient adaptive proposals can outperform basic MC even with fewer samples.
        /// - Variance reduction via adaptation leads to more precise integral estimates.
        /// </summary>
        /// <remarks>
        ///     <list type="bullet">
        ///         <item>
        ///         Follows the "moment‑matching Population Monte‑Carlo" framework of Cappé et al. (2008, Ann. Stat.).
        ///         Weighted samples update the mean and variance of a truncated‑Normal component that focuses on the overlap hotspot.
        ///         </item>
        ///     </list>
        /// </remarks>
        [TestMethod]
        public void Test_Adaptive_ImportanceSampling_Integration_ReducedSamples()
        {
            // Adaptation phase settings 
            const int adaptiveIterations = 5;     // Number of iterations to update proposal distribution
            const int adaptiveSamples = 200;      // Number of samples per iteration (for computing weighted stats)
            const double alpha = 0.75;            // Smoothing parameter for proposal update (exponential moving average)
            const double eps = 1e-6;        // Minimum variance to prevent numerical collapse

            // Define integration domain over [1,21]^2 (representing storm transposition space)
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin); // Domain area = 400

            // Final integration sample count 
            int N = 10000;
            var prng = new MersenneTwister(12345);  // RNG with fixed seed for reproducibility

            // Target distribution: uniform over the integration domain
            var xDist = new Uniform(xmin, xmax);
            var yDist = new Uniform(ymin, ymax);

            // Initial proposal distribution
            // Starts far from the watershed to simulate realistic initial guess
            double muX = 7.0, muY = 1.0;
            double varX = 30.0, varY = 30.0;

            // Initialize with somewhat arbitrary truncated normals
            var xISDist = new TruncatedNormal(muX, Math.Sqrt(varX), xmin, xmax);
            var yISDist = new TruncatedNormal(muY, Math.Sqrt(varY), ymin, ymax);

            // Adaptive loop: iteratively refine the proposal distribution
            for (int t = 0; t < adaptiveIterations; t++)
            {
                // Arrays to hold samples and corresponding weights
                double[] xs = new double[adaptiveSamples];
                double[] ys = new double[adaptiveSamples];
                double[] w = new double[adaptiveSamples];

                double sumW = 0.0;

                // 1. Sample from current proposal and compute normalized weights
                for (int i = 0; i < adaptiveSamples; i++)
                {
                    double x = xISDist.InverseCDF(prng.NextDouble());
                    double y = yISDist.InverseCDF(prng.NextDouble());
                    xs[i] = x;
                    ys[i] = y;

                    double p = xDist.PDF(x) * yDist.PDF(y);       // Uniform target density
                    double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal (truncated normal)
                    double wi = (q > 0.0) ? p / q : 0.0;

                    w[i] = SST_Toy_Integrand(x, y) * wi;  // Weighted contribution only where inside watershed
                    sumW += w[i];
                }

                // 2. Normalize weights so they sum to 1 (acts as discrete PDF for moment matching)
                if (sumW <= 0) sumW = 1e-16;  // Prevent division by zero
                for (int i = 0; i < adaptiveSamples; i++)
                    w[i] /= sumW;

                // 3. Compute weighted means (first moments)
                double newMuX = 0.0, newMuY = 0.0;
                for (int i = 0; i < adaptiveSamples; i++)
                {
                    newMuX += w[i] * xs[i];
                    newMuY += w[i] * ys[i];
                }

                // 4. Compute weighted variances (second moments)
                double newVarX = eps, newVarY = eps;
                for (int i = 0; i < adaptiveSamples; i++)
                {
                    double dx = xs[i] - newMuX;
                    double dy = ys[i] - newMuY;
                    newVarX += w[i] * dx * dx;
                    newVarY += w[i] * dy * dy;
                }

                // 5. Apply exponential smoothing to stabilize updates
                muX = alpha * newMuX + (1 - alpha) * muX;
                muY = alpha * newMuY + (1 - alpha) * muY;
                varX = alpha * newVarX + (1 - alpha) * varX;
                varY = alpha * newVarY + (1 - alpha) * varY;

                // 6. Update the proposal distribution using new smoothed parameters
                xISDist = new TruncatedNormal(muX, Math.Sqrt(varX), xmin, xmax);
                yISDist = new TruncatedNormal(muY, Math.Sqrt(varY), ymin, ymax);
            }

            // Final integration step using adapted proposal
            double sum = 0, sum2 = 0;

            for (int i = 0; i < N; i++)
            {
                double x = xISDist.InverseCDF(prng.NextDouble());
                double y = yISDist.InverseCDF(prng.NextDouble());

                double p = xDist.PDF(x) * yDist.PDF(y);
                double q = xISDist.PDF(x) * yISDist.PDF(y);
                double weight = (q > 0.0) ? p / q : 0.0;

                double f = SST_Toy_Integrand(x, y) * weight;

                sum += f;
                sum2 += f * f;
            }

            double avg = sum / N;
            double avg2 = sum2 / N;

            // Estimate integral by multiplying average value by domain area
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Example Output:
            // Mean ≈ 8.96
            // Standard Error ≈ 0.076
            // Demonstrates strong convergence even from a poor initial guess
        }

        #endregion

        #region SST Toy Parametric Simulation Problem

        /* 
        * Motivation:
        * Now that we’ve shown how importance sampling improves integration accuracy for spatial storm transposition,
        * we move on to a more realistic simulation-based SST problem:
        * - Storms are randomly transposed over a domain.
        * - Storms have variable sizes (represented as square blocks).
        * - If a storm overlaps the watershed, a rainfall depth is drawn from a normal distribution.
        * 
        * Goal: Simulate a basin-averaged frequency curve using naive Monte Carlo sampling.
        */

        /// <summary>
        /// Struct representing a storm cell for this toy problem. 
        /// </summary>
        public struct StormCell
        {
            public double SideLength;      // km  (we keep the square idealization for now)
            public double MeanDepth;       // mm  (uniform across the footprint – step‑1 realism)
        }

        /// <summary>
        /// Storm catalog – lengths roughly 2, 4, 6 km and realistic depths
        /// </summary>
        private readonly StormCell[] _stormCatalogue = new[]
        {
            new StormCell { SideLength = 2.0, MeanDepth = 45 },   // convective cell
            new StormCell { SideLength = 4.0, MeanDepth = 80 },   // meso‑scale core
            new StormCell { SideLength = 6.0, MeanDepth = 120 }   // large frontal band
        };

        /// <summary>
        /// Toy simulation for stochastic storm transposition (SST) with storm-watershed overlap.
        /// If the storm overlaps the watershed, a rainfall depth is sampled from ~N(10,2).
        /// </summary>
        /// <param name="x">X-coordinate of storm centroid</param>
        /// <param name="y">Y-coordinate of storm centroid</param>
        /// <param name="stormIndex">Index to select storm shape from array of square sizes</param>
        /// <param name="rnd">Random uniform(0,1) number used to sample from storm depth distribution</param>
        /// <returns>Storm depth over watershed (0 if no overlap)</returns>
        private double SST_Toy_Parametric_Simulator(double x, double y, int stormIndex, double rnd)
        {
            // Watershed definition: 3x3 square centered at (5.5,4.5)
            double ax = 5.5, ay = 4.5;
            double aSize = 3;

            // Compute watershed boundaries
            double aMinX = ax - aSize / 2;
            double aMaxX = ax + aSize / 2;
            double aMinY = ay - aSize / 2;
            double aMaxY = ay + aSize / 2;

            // Compute storm boundaries based on storm size and centroid
            double bMinX = x - _stormCatalogue[stormIndex].SideLength / 2;
            double bMaxX = x + _stormCatalogue[stormIndex].SideLength / 2;
            double bMinY = y - _stormCatalogue[stormIndex].SideLength / 2;
            double bMaxY = y + _stormCatalogue[stormIndex].SideLength / 2;

            // Check for bounding-box overlap (axis-aligned square intersection)
            bool overlap = !(aMaxX < bMinX || aMinX > bMaxX || aMaxY < bMinY || aMinY > bMaxY);

            if (overlap)
            {
                // Storm intersects the watershed – sample a rainfall depth
                var dist = new Normal(10, 2);  // Gaussian storm depth distribution
                return Math.Max(0, dist.InverseCDF(rnd)); // Ensure non-negative depth
            }

            // No overlap: return zero depth
            return 0;
        }

        /// <summary>
        /// Test basic Monte Carlo simulation with 100,000 samples.
        /// </summary>
        [TestMethod]
        public void Test_BasicMonteCarlo_Parametric_Simulation()
        {
            // Define simulation domain (storm centroids fall within [1,21] × [1,21])
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin); // Domain area = 400

            const int N = 100000;  // Number of simulated storms

            // Random number generators for:
            // storm location (xy), shape (index), and depth (uniform sampling from CDF)
            var prngXY = new MersenneTwister(12345);
            var prngStorm = new MersenneTwister(91011);
            var prngDepth = new MersenneTwister(45678);

            // Uniform distributions for x and y location
            var xDist = new Uniform(xmin, xmax);
            var yDist = new Uniform(ymin, ymax);

            // Store sampled depths for plotting and diagnostics
            var depths = new double[N];

            double sum = 0, sum2 = 0;

            // Perform Monte Carlo simulation
            for (int i = 0; i < N; i++)
            {
                // Sample storm centroid
                double x = xDist.InverseCDF(prngXY.NextDouble());
                double y = yDist.InverseCDF(prngXY.NextDouble());

                // Randomly choose a storm size (0, 1, or 2 corresponding to size 2, 4, 6)
                int stormIndex = prngStorm.Next(0, 3);

                // Sample a storm depth (0 if not overlapping)
                depths[i] = SST_Toy_Parametric_Simulator(x, y, stormIndex, prngDepth.NextDouble());

                // Accumulate results for mean and variance estimation
                sum += depths[i];
                sum2 += depths[i] * depths[i];
            }

            double avg = sum / N;
            double avg2 = sum2 / N;
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output (N=100,000): Mean ≈ 486.55, SE ≈ 4.23 (depends on overlap fraction and depth distribution)

            // Sort depths to compute frequency curve
            Array.Sort(depths);
            var pp = PlottingPositions.Weibull(N);

            // Print results for plotting depth-frequency curve
            for (int i = 0; i < N; i++)
            {
                Debug.WriteLine(depths[i] + "," + pp[i]);
            }

            // This frequency curve can now be plotted as:
            // x-axis: Return Period = 1 / (1 - pp)
            // y-axis: Depth
        }

        /// <summary>
        /// Importance Sampling SST simulation with 10,000 samples.
        /// Improves precision by sampling more often near the watershed using a Truncated Normal distribution.
        /// Weights are used to correct the bias from sampling non-uniformly.
        /// </summary>
        [TestMethod]
        public void Test_ImportanceSampling_Parametric_Simulation_Reduced_Samples()
        {
            // Define spatial domain where storm centroids are sampled
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin); // Domain area = 400

            const int N = 10000; // Reduced number of samples due to improved efficiency

            // Random number generators for each random component
            var prngXY = new MersenneTwister(12345);   // For x-y storm centroids
            var prngStorm = new MersenneTwister(91011); // For storm size index
            var prngDepth = new MersenneTwister(45678); // For depth sampling

            // Define target (uniform) and proposal (importance sampling) distributions
            var xDist = new Uniform(xmin, xmax); // Target distribution (uniform)
            var yDist = new Uniform(ymin, ymax);
            var xISDist = new TruncatedNormal(5.5, 5.8, xmin, xmax); // Proposal focused on watershed
            var yISDist = new TruncatedNormal(4.5, 5.8, ymin, ymax);

            // The challenge with this approach is that we must hand-tune the importance distribution parameters.
            // This is not a viable approach for widespread use of SST

            // Allocate storage for results
            var depths = new double[N];  // Simulated rainfall depths
            var w = new double[N];       // Importance sampling weights

            double sum = 0.0, sum2 = 0.0;
            double sumW = 0.0;

            // Perform importance-weighted simulation 
            for (int i = 0; i < N; i++)
            {
                // Sample storm centroid from proposal distribution
                double x = xISDist.InverseCDF(prngXY.NextDouble());
                double y = yISDist.InverseCDF(prngXY.NextDouble());

                // Sample storm shape (square side: 2, 4, or 6)
                int stormIndex = prngStorm.Next(0, 3);

                // Simulate depth based on overlap and storm randomness
                depths[i] = SST_Toy_Parametric_Simulator(x, y, stormIndex, prngDepth.NextDouble());

                // Compute importance sampling weight: p(x,y) / q(x,y)
                double p = xDist.PDF(x) * yDist.PDF(y);       // Uniform PDF
                double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal PDF
                w[i] = (q > 0.0) ? p / q : 0.0;
                sumW += w[i];

                // Apply weight to response
                double f = depths[i] * w[i];

                sum += f;
                sum2 += f * f;
            }

            // Estimate mean and standard error of basin-average rainfall
            double avg = sum / N;
            double avg2 = sum2 / N;
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output (N=10,000): Mean ≈ 487.93, SE ≈ 6.81 (depends on overlap fraction and depth distribution)
            // The standard error is slightly larger than the basic Monte Carlo runs with 100,000 samples

            // Normalize weights for frequency curve plotting
            for (int i = 0; i < N; i++)
                w[i] /= sumW;

            // Sort depths and weights together
            Array.Sort(depths, w); // Sorts depths and keeps weights aligned

            // Compute weighted plotting positions (cumulative sum of normalized weights)
            var pp = new double[N];
            pp[0] = w[0];
            for (int i = 1; i < N; i++)
                pp[i] = pp[i - 1] + w[i];

            // Print results for plotting depth-frequency curve
            for (int i = 0; i < N; i++)
            {
                Debug.WriteLine(depths[i] + "," + pp[i]);
            }

            // This frequency curve can now be plotted as:
            // x-axis: Return Period = 1 / (1 - pp)
            // y-axis: Depth
        }

        /// <summary>
        /// Adaptive Importance Sampling SST Parametric Simulation (10,000 samples).
        ///
        /// This simulates storm transpositions where the storm centroid is sampled from
        /// a learned importance distribution. The distribution is adapted over several
        /// iterations using weighted samples to target regions that overlap the watershed.
        ///
        /// Key Features:
        /// - Adaptive proposal centered based on weighted SST depth realizations
        /// - Supports multiple storm sizes
        /// - Generates a weighted depth-frequency curve using importance sampling
        /// </summary>
        /// <remarks>
        ///     <list type="bullet">
        ///         <item>
        ///         Follows the "moment‑matching Population Monte‑Carlo" framework of Cappé et al. (2008, Ann. Stat.).
        ///         Weighted samples update the mean and variance of a *narrow* truncated‑Normal component that
        ///         focuses on the overlap hotspot, while a *wide* component (variance ≈ Var[U(1,20)]) preserves global
        ///         support and keeps importance weights finite.
        ///         </item>
        ///         <item>
        ///         The mixture weight (beta) is learned by a damped EM step (Bugallo et al., 2017, IEEE SPM 34‑4).
        ///         </item>
        ///         <item> 
        ///         Storm size is discrete (side = 2,4,6) and depth is produced by SST_Toy_Parametric_Simulator().
        ///         </item>
        ///     </list>
        /// </remarks>
        [TestMethod]
        public void Test_Adaptive_ImportanceSampling_Parametric_Simulation_ReducedSamples()
        {
            // Adaptation phase settings 
            const int adaptiveCycles = 5;                  // Number of cycles to update proposal distribution
            const int adaptiveSamples = 400;                // Number of samples per cycle (for computing weighted stats)
            const double alpha = 0.75;                      // Exponential smoothing parameter for proposal update (exponential moving average)
            const double eps = 1e-6;                        // Minimum variance to prevent numerical collapse
            const double betaFloor = 0.01, betaCeil = 0.5;  // Bounds on the mixing weight (watershed is at least 1% of domain, and not larger than 50%)
            const double lambda = 0.3;                      // learning rate for the mixing weight

            // Define integration domain over [1,21]^2 (representing storm transposition space)
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin); // Domain area = 400

            // Wide variance equal to Var[U(1,21)] = (20)²/12 ≈ 33.33 (kept fixed)
            const double xWideVar = (xmax - xmin) * (xmax - xmin) / 12.0;
            const double yWideVar = (ymax - ymin) * (ymax - ymin) / 12.0;

            // Final integration sample count 
            int N = 10000;

            // Random number generators for each random component
            var prngXY = new MersenneTwister(12345);   // For x-y storm centroids
            var prngStorm = new MersenneTwister(91011); // For storm size index
            var prngDepth = new MersenneTwister(45678); // For depth sampling

            // Target distribution: uniform over the integration domain
            var xDist = new Uniform(xmin, xmax);
            var yDist = new Uniform(ymin, ymax);

            // Initial proposal distribution parameters
            // These parameters are learned in the adaption phase
            double muX = 7.0, muY = 1.0; // Deliberately off-center
            double varX = 50.0, varY = 50.0; // Deliberately wide
            double beta = 0.25; // Somewhat close to the target

            // The initial proposals are mixtures of truncated Normal distributions.
            // One distribution given the weight of beta is allowed to home in on the mean and variance of the watershed
            // The second distribution centers on the watershed but the variance is kept wide to provide coverage of all the events
            // that fall outside the watershed and contribute nothing to the basin-average.
            var xISDist = new Mixture(new double[] { beta, 1 - beta }, new IUnivariateDistribution[] { new TruncatedNormal(muX, Math.Sqrt(varX), xmin, xmax), new TruncatedNormal(muX, xWideVar, xmin, xmax) });
            var yISDist = new Mixture(new double[] { beta, 1 - beta }, new IUnivariateDistribution[] { new TruncatedNormal(muY, Math.Sqrt(varY), ymin, ymax), new TruncatedNormal(muY, yWideVar, ymin, ymax) });

            // Adaptive loop: iteratively refine the proposal distribution
            for (int t = 0; t < adaptiveCycles; t++)
            {
                // Arrays to hold samples and corresponding weights
                double[] xs = new double[adaptiveSamples];
                double[] ys = new double[adaptiveSamples];
                var weights = new double[adaptiveSamples];
                var depthHit = new double[adaptiveSamples];
                var hits = new int[adaptiveSamples];
                double sumW = 0;

                // 1. Sample from current proposal and compute normalized weights
                for (int i = 0; i < adaptiveSamples; i++)
                {
                    // Sample storm centroid from proposal distribution
                    double x = xISDist.InverseCDF(prngXY.NextDouble());
                    double y = yISDist.InverseCDF(prngXY.NextDouble());
                    xs[i] = x;
                    ys[i] = y;

                    // Sample storm shape (square side: 2, 4, or 6)
                    int stormIndex = prngStorm.Next(0, 3);

                    // Simulate depth based on overlap and storm randomness
                    depthHit[i] = SST_Toy_Parametric_Simulator(x, y, stormIndex, prngDepth.NextDouble());
                    hits[i] = depthHit[i] > 0 ? 1 : 0;

                    // Compute importance sampling weight: p(x,y) / q(x,y)
                    double p = xDist.PDF(x) * yDist.PDF(y);       // Uniform PDF
                    double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal PDF
                    double wi = (q > 0.0) ? p / q : 0.0;
                    weights[i] = wi;
                    sumW += weights[i];           
                }

               
                // 2. Mixing weight update: expected hit‑probability under geometry weights
                double pHit = xs.Select((_, i) => weights[i] * hits[i]).Sum() / sumW;  // E_p/q[hit]
                double betaNew = Math.Clamp((1 - lambda) * beta + lambda * pHit, betaFloor, betaCeil);
                beta = betaNew;

                // 3-5.  compute µ, σ² from *hits only*
                var idxHit = Enumerable.Range(0, adaptiveSamples).Where(i => hits[i] == 1).ToArray();
                if (idxHit.Length > 10)                    // need at least a few hits
                {
                    // temper the weights so no single point dominates
                    const double K = 1.0;                  // mm of smoothing
                    var wHit = idxHit.Select(i => weights[i] * (depthHit[i] + K)).ToArray();

                    // 2. Normalize weights so they sum to 1 (acts as discrete PDF for moment matching)
                    double sumWH = wHit.Sum();
                    if (sumWH <= 0) sumWH = 1e-16;  // Prevent division by zero
                    for (int k = 0; k < wHit.Length; k++) 
                        wHit[k] /= sumWH;

                    var xHit = idxHit.Select(i => xs[i]).ToArray();
                    var yHit = idxHit.Select(i => ys[i]).ToArray();

                    // 3-4. Moment matching (Cappé et al. 2008)
                    // 3. Compute weighted means (first moments)
                    double newMuX = 0.0, newMuY = 0.0;
                    for (int i = 0; i < idxHit.Length; i++)
                    {
                        newMuX += wHit[i] * xHit[i];
                        newMuY += wHit[i] * yHit[i];
                    }

                    // 4. Compute weighted variances (second moments)
                    double newVarX = eps, newVarY = eps;
                    for (int i = 0; i < idxHit.Length; i++)
                    {
                        double dx = xHit[i] - newMuX;
                        double dy = yHit[i] - newMuY;
                        newVarX += wHit[i] * dx * dx;
                        newVarY += wHit[i] * dy * dy;
                    }

                    // 5. Apply exponential smoothing to stabilize updates
                    muX = alpha * newMuX + (1 - alpha) * muX;
                    muY = alpha * newMuY + (1 - alpha) * muY;
                    varX = alpha * newVarX + (1 - alpha) * varX;
                    varY = alpha * newVarY + (1 - alpha) * varY;
                }

                // 6. Update the proposal distribution using new parameters
                xISDist = new Mixture(new double[] { beta, 1 - beta }, new IUnivariateDistribution[] { new TruncatedNormal(muX, Math.Sqrt(varX), xmin, xmax), new TruncatedNormal(muX, xWideVar, xmin, xmax) });
                yISDist = new Mixture(new double[] { beta, 1 - beta }, new IUnivariateDistribution[] { new TruncatedNormal(muY, Math.Sqrt(varY), ymin, ymax), new TruncatedNormal(muY, yWideVar, ymin, ymax) });
            }

            // Allocate storage for results
            var depths = new double[N];  // Simulated rainfall depths
            var w = new double[N];       // Importance sampling weights

            double sum = 0.0, sum2 = 0.0, sumWFinal = 0.0;

            // Perform importance-weighted simulation 
            for (int i = 0; i < N; i++)
            {
                // Sample storm centroid from proposal distribution
                double x = xISDist.InverseCDF(prngXY.NextDouble());
                double y = yISDist.InverseCDF(prngXY.NextDouble());

                // Sample storm shape (square side: 2, 4, or 6)
                int stormIndex = prngStorm.Next(0, 3);

                // Simulate depth based on overlap and storm randomness
                depths[i] = SST_Toy_Parametric_Simulator(x, y, stormIndex, prngDepth.NextDouble());

                // Compute importance sampling weight: p(x,y) / q(x,y)
                double p = xDist.PDF(x) * yDist.PDF(y);       // Uniform PDF
                double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal PDF
                w[i] = (q > 0.0) ? p / q : 0.0;
                sumWFinal += w[i];

                // Apply weight to response
                double f = depths[i] * w[i];

                sum += f;
                sum2 += f * f;
            }

            // Estimate mean and standard error of basin-average rainfall
            double avg = sum / N;
            double avg2 = sum2 / N;
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output (N=10,000): Mean ≈ 501.07, SE ≈ 10.57 (depends on overlap fraction and depth distribution)

            // Normalize weights for frequency curve plotting
            for (int i = 0; i < N; i++)
                w[i] /= sumWFinal;

            // Sort depths and weights together
            Array.Sort(depths, w); // Sorts depths and keeps weights aligned

            // Compute weighted plotting positions (cumulative sum of normalized weights)
            var pp = new double[N];
            pp[0] = w[0];
            for (int i = 1; i < N; i++)
                pp[i] = pp[i - 1] + w[i];

            // Print results for plotting depth-frequency curve
            for (int i = 0; i < N; i++)
            {
                Debug.WriteLine(depths[i] + "," + pp[i]);
            }

            // This frequency curve can now be plotted as:
            // x-axis: Return Period = 1 / (1 - pp)
            // y-axis: Depth
        }

        #endregion

        #region SST Toy Nonparametric Simulation Problem


        /// <summary>
        /// Basin‑average precipitation depth (mm) from a square storm patch.
        /// Returns 0 mm when the storm does not intersect the basin.
        /// </summary>
        /// <param name="x">Storm‑centroid x‑coordinate (km)</param>
        /// <param name="y">Storm‑centroid y‑coordinate (km)</param>
        /// <param name="stormIndex">Index into _stormCatalogue</param>
        private double SST_Toy_Nonparametric_Simulator(double x, double y, int stormIndex)
        {
            // -------------- Basin definition (still a 3 × 3 km square) -------------
            const double basinCentroidX = 5.5, basinCentroidY = 4.5;  // km
            const double basinSide = 3.0;                        // km
            double basinArea = basinSide * basinSide;      // km²

            double basinMinX = basinCentroidX - basinSide / 2.0;
            double basinMaxX = basinCentroidX + basinSide / 2.0;
            double basinMinY = basinCentroidY - basinSide / 2.0;
            double basinMaxY = basinCentroidY + basinSide / 2.0;

            // ------------------------ Storm geometry -------------------------------
            StormCell sc = _stormCatalogue[stormIndex];
            double sMinX = x - sc.SideLength / 2.0;
            double sMaxX = x + sc.SideLength / 2.0;
            double sMinY = y - sc.SideLength / 2.0;
            double sMaxY = y + sc.SideLength / 2.0;

            // ---------------- Overlap rectangle (if any) ---------------------------
            double overlapX = Math.Max(0.0, Math.Min(basinMaxX, sMaxX) - Math.Max(basinMinX, sMinX));
            double overlapY = Math.Max(0.0, Math.Min(basinMaxY, sMaxY) - Math.Max(basinMinY, sMinY));
            double overlapArea = overlapX * overlapY;                // km²

            if (overlapArea <= 0.0) return 0.0;                      // miss –> 0 mm

            // -------------- Basin‑average precipitation (uniform depth) ------------
            double precipAvg = (overlapArea / basinArea) * sc.MeanDepth;
            return precipAvg;                                        // mm
        }


        /// <summary>
        /// Test basic Monte Carlo simulation with 100,000 samples.
        /// </summary>
        [TestMethod]
        public void Test_BasicMonteCarlo_Nonparametric_Simulation()
        {
            // Define simulation domain (storm centroids fall within [1,21] × [1,21])
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin); // Domain area = 400

            const int N = 100000;  // Number of simulated storms

            // Random number generators for:
            // storm location (xy), shape (index), and depth (uniform sampling from CDF)
            var prngXY = new MersenneTwister(12345);
            var prngStorm = new MersenneTwister(91011);

            // Uniform distributions for x and y location
            var xDist = new Uniform(xmin, xmax);
            var yDist = new Uniform(ymin, ymax);

            // Store sampled depths for plotting and diagnostics
            var depths = new double[N];

            double sum = 0, sum2 = 0;

            // Perform Monte Carlo simulation
            for (int i = 0; i < N; i++)
            {
                // Sample storm centroid
                double x = xDist.InverseCDF(prngXY.NextDouble());
                double y = yDist.InverseCDF(prngXY.NextDouble());

                // Randomly choose a storm size (0, 1, or 2 corresponding to size 2, 4, 6)
                int stormIndex = prngStorm.Next(0, 3);

                // Sample a storm depth (0 if not overlapping)
                depths[i] = SST_Toy_Nonparametric_Simulator(x, y, stormIndex);

                // Accumulate results for mean and variance estimation
                sum += depths[i];
                sum2 += depths[i] * depths[i];
            }

            double avg = sum / N;
            double avg2 = sum2 / N;
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output (N=100,000): Mean ≈ 1880.06, SE ≈ 22.80 (depends on overlap fraction and depth distribution)

            // Sort depths to compute frequency curve
            Array.Sort(depths);
            var pp = PlottingPositions.Weibull(N);

            // Print results for plotting depth-frequency curve
            for (int i = 0; i < N; i++)
            {
                Debug.WriteLine(depths[i] + "," + pp[i]);
            }

            // This frequency curve can now be plotted as:
            // x-axis: Return Period = 1 / (1 - pp)
            // y-axis: Depth
        }

        /// <summary>
        /// Importance Sampling SST simulation with 10,000 samples.
        /// Improves precision by sampling more often near the watershed using a Truncated Normal distribution.
        /// Weights are used to correct the bias from sampling non-uniformly.
        /// </summary>
        [TestMethod]
        public void Test_ImportanceSampling_Nonparametric_Simulation_Reduced_Samples()
        {
            // Define spatial domain where storm centroids are sampled
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin); // Domain area = 400

            const int N = 10000; // Reduced number of samples due to improved efficiency

            // Random number generators for each random component
            var prngXY = new MersenneTwister(12345);   // For x-y storm centroids
            var prngStorm = new MersenneTwister(91011); // For storm size index

            // Define target (uniform) and proposal (importance sampling) distributions
            var xDist = new Uniform(xmin, xmax); // Target distribution (uniform)
            var yDist = new Uniform(ymin, ymax);
            var xISDist = new TruncatedNormal(5.5, 5.8, xmin, xmax); // Proposal focused on watershed
            var yISDist = new TruncatedNormal(4.5, 5.8, ymin, ymax);

            // The challenge with this approach is that we must hand-tune the importance distribution parameters.
            // This is not a viable approach for widespread use of SST

            // Allocate storage for results
            var depths = new double[N];  // Simulated rainfall depths
            var w = new double[N];       // Importance sampling weights

            double sum = 0.0, sum2 = 0.0;
            double sumW = 0.0;

            // Perform importance-weighted simulation 
            for (int i = 0; i < N; i++)
            {
                // Sample storm centroid from proposal distribution
                double x = xISDist.InverseCDF(prngXY.NextDouble());
                double y = yISDist.InverseCDF(prngXY.NextDouble());

                // Sample storm shape (square side: 2, 4, or 6)
                int stormIndex = prngStorm.Next(0, 3);

                // Simulate depth based on overlap and storm randomness
                depths[i] = SST_Toy_Nonparametric_Simulator(x, y, stormIndex);

                // Compute importance sampling weight: p(x,y) / q(x,y)
                double p = xDist.PDF(x) * yDist.PDF(y);       // Uniform PDF
                double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal PDF
                w[i] = (q > 0.0) ? p / q : 0.0;
                sumW += w[i];

                // Apply weight to response
                double f = depths[i] * w[i];

                sum += f;
                sum2 += f * f;
            }

            // Estimate mean and standard error of basin-average rainfall
            double avg = sum / N;
            double avg2 = sum2 / N;
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output (N=10,000): Mean ≈ 1850.20, SE ≈ 37.16 (depends on overlap fraction and depth distribution)
            // The standard error is slightly larger than the basic Monte Carlo runs with 100,000 samples

            // Normalize weights for frequency curve plotting
            for (int i = 0; i < N; i++)
                w[i] /= sumW;

            // Sort depths and weights together
            Array.Sort(depths, w); // Sorts depths and keeps weights aligned

            // Compute weighted plotting positions (cumulative sum of normalized weights)
            var pp = new double[N];
            pp[0] = w[0];
            for (int i = 1; i < N; i++)
                pp[i] = pp[i - 1] + w[i];

            // Print results for plotting depth-frequency curve
            for (int i = 0; i < N; i++)
            {
                Debug.WriteLine(depths[i] + "," + pp[i]);
            }

            // This frequency curve can now be plotted as:
            // x-axis: Return Period = 1 / (1 - pp)
            // y-axis: Depth
        }

        /// <summary>
        /// Adaptive Importance Sampling SST Nonparametric Simulation (10,000 samples).
        ///
        /// This simulates storm transpositions where the storm centroid is sampled from
        /// a learned importance distribution. The distribution is adapted over several
        /// iterations using weighted samples to target regions that overlap the watershed.
        ///
        /// Key Features:
        /// - Adaptive proposal centered based on weighted SST depth realizations
        /// - Supports multiple storm sizes
        /// - Generates a weighted depth-frequency curve using importance sampling
        /// </summary>
        /// <remarks>
        ///     <list type="bullet">
        ///         <item>
        ///         Follows the "moment‑matching Population Monte‑Carlo" framework of Cappé et al. (2008, Ann. Stat.).
        ///         Weighted samples update the mean and variance of a *narrow* truncated‑Normal component that
        ///         focuses on the overlap hotspot, while a *wide* component (variance ≈ Var[U(1,20)]) preserves global
        ///         support and keeps importance weights finite.
        ///         </item>
        ///         <item>
        ///         The mixture weight (beta) is learned by a damped EM step (Bugallo et al., 2017, IEEE SPM 34‑4).
        ///         </item>
        ///         <item> 
        ///         Storm size is discrete (side = 2,4,6) and depth is produced by SST_Toy_Parametric_Simulator().
        ///         </item>
        ///     </list>
        /// </remarks>
        [TestMethod]
        public void Test_Adaptive_ImportanceSampling_Nonparametric_Simulation_ReducedSamples()
        {
            // Adaptation phase settings 
            const int adaptiveCycles = 5;                  // Number of cycles to update proposal distribution
            const int adaptiveSamples = 400;                // Number of samples per cycle (for computing weighted stats)
            const double alpha = 0.75;                      // Exponential smoothing parameter for proposal update (exponential moving average)
            const double eps = 1e-6;                        // Minimum variance to prevent numerical collapse
            const double betaFloor = 0.01, betaCeil = 0.5;  // Bounds on the mixing weight (watershed is at least 1% of domain, and not larger than 50%)
            const double lambda = 0.3;                      // learning rate for the mixing weight

            // Define integration domain over [1,21]^2 (representing storm transposition space)
            const double xmin = 1.0, xmax = 21.0;
            const double ymin = 1.0, ymax = 21.0;
            double domainArea = (xmax - xmin) * (ymax - ymin); // Domain area = 400

            // Wide variance equal to Var[U(1,21)] = (20)²/12 ≈ 33.33 (kept fixed)
            const double xWideVar = (xmax - xmin) * (xmax - xmin) / 12.0;
            const double yWideVar = (ymax - ymin) * (ymax - ymin) / 12.0;

            // Final integration sample count 
            int N = 10000;

            // Random number generators for each random component
            var prngXY = new MersenneTwister(12345);   // For x-y storm centroids
            var prngStorm = new MersenneTwister(91011); // For storm size index

            // Target distribution: uniform over the integration domain
            var xDist = new Uniform(xmin, xmax);
            var yDist = new Uniform(ymin, ymax);

            // Initial proposal distribution parameters
            // These parameters are learned in the adaption phase
            double muX = 7.0, muY = 1.0; // Deliberately off-center
            double varX = 50.0, varY = 50.0; // Deliberately wide
            double beta = 0.25; // Somewhat close to the target

            // The initial proposals are mixtures of truncated Normal distributions.
            // One distribution given the weight of beta is allowed to home in on the mean and variance of the watershed
            // The second distribution centers on the watershed but the variance is kept wide to provide coverage of all the events
            // that fall outside the watershed and contribute nothing to the basin-average.
            var xISDist = new Mixture(new double[] { beta, 1 - beta }, new IUnivariateDistribution[] { new TruncatedNormal(muX, Math.Sqrt(varX), xmin, xmax), new TruncatedNormal(muX, xWideVar, xmin, xmax) });
            var yISDist = new Mixture(new double[] { beta, 1 - beta }, new IUnivariateDistribution[] { new TruncatedNormal(muY, Math.Sqrt(varY), ymin, ymax), new TruncatedNormal(muY, yWideVar, ymin, ymax) });

            // Adaptive loop: iteratively refine the proposal distribution
            for (int t = 0; t < adaptiveCycles; t++)
            {
                // Arrays to hold samples and corresponding weights
                double[] xs = new double[adaptiveSamples];
                double[] ys = new double[adaptiveSamples];
                var weights = new double[adaptiveSamples];
                var depthHit = new double[adaptiveSamples];
                var hits = new int[adaptiveSamples];
                double sumW = 0;

                // 1. Sample from current proposal and compute normalized weights
                for (int i = 0; i < adaptiveSamples; i++)
                {
                    // Sample storm centroid from proposal distribution
                    double x = xISDist.InverseCDF(prngXY.NextDouble());
                    double y = yISDist.InverseCDF(prngXY.NextDouble());
                    xs[i] = x;
                    ys[i] = y;

                    // Sample storm shape (square side: 2, 4, or 6)
                    int stormIndex = prngStorm.Next(0, 3);

                    // Simulate depth based on overlap and storm randomness
                    depthHit[i] = SST_Toy_Nonparametric_Simulator(x, y, stormIndex);
                    hits[i] = depthHit[i] > 0 ? 1 : 0;

                    // Compute importance sampling weight: p(x,y) / q(x,y)
                    double p = xDist.PDF(x) * yDist.PDF(y);       // Uniform PDF
                    double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal PDF
                    double wi = (q > 0.0) ? p / q : 0.0;
                    weights[i] = wi;
                    sumW += weights[i];
                }


                // 2. Mixing weight update: expected hit‑probability under geometry weights
                double pHit = xs.Select((_, i) => weights[i] * hits[i]).Sum() / sumW;  // E_p/q[hit]
                double betaNew = Math.Clamp((1 - lambda) * beta + lambda * pHit, betaFloor, betaCeil);
                beta = betaNew;

                // 3-5.  compute µ, σ² from *hits only*
                var idxHit = Enumerable.Range(0, adaptiveSamples).Where(i => hits[i] == 1).ToArray();
                if (idxHit.Length > 10)                    // need at least a few hits
                {
                    // temper the weights so no single point dominates
                    const double K = 1.0;                  // mm of smoothing
                    var wHit = idxHit.Select(i => weights[i] * (depthHit[i] + K)).ToArray();

                    // 2. Normalize weights so they sum to 1 (acts as discrete PDF for moment matching)
                    double sumWH = wHit.Sum();
                    if (sumWH <= 0) sumWH = 1e-16;  // Prevent division by zero
                    for (int k = 0; k < wHit.Length; k++)
                        wHit[k] /= sumWH;

                    var xHit = idxHit.Select(i => xs[i]).ToArray();
                    var yHit = idxHit.Select(i => ys[i]).ToArray();

                    // 3-4. Moment matching (Cappé et al. 2008)
                    // 3. Compute weighted means (first moments)
                    double newMuX = 0.0, newMuY = 0.0;
                    for (int i = 0; i < idxHit.Length; i++)
                    {
                        newMuX += wHit[i] * xHit[i];
                        newMuY += wHit[i] * yHit[i];
                    }

                    // 4. Compute weighted variances (second moments)
                    double newVarX = eps, newVarY = eps;
                    for (int i = 0; i < idxHit.Length; i++)
                    {
                        double dx = xHit[i] - newMuX;
                        double dy = yHit[i] - newMuY;
                        newVarX += wHit[i] * dx * dx;
                        newVarY += wHit[i] * dy * dy;
                    }

                    // 5. Apply exponential smoothing to stabilize updates
                    muX = alpha * newMuX + (1 - alpha) * muX;
                    muY = alpha * newMuY + (1 - alpha) * muY;
                    varX = alpha * newVarX + (1 - alpha) * varX;
                    varY = alpha * newVarY + (1 - alpha) * varY;
                }

                // 6. Update the proposal distribution using new parameters
                xISDist = new Mixture(new double[] { beta, 1 - beta }, new IUnivariateDistribution[] { new TruncatedNormal(muX, Math.Sqrt(varX), xmin, xmax), new TruncatedNormal(muX, xWideVar, xmin, xmax) });
                yISDist = new Mixture(new double[] { beta, 1 - beta }, new IUnivariateDistribution[] { new TruncatedNormal(muY, Math.Sqrt(varY), ymin, ymax), new TruncatedNormal(muY, yWideVar, ymin, ymax) });
            }

            // Allocate storage for results
            var depths = new double[N];  // Simulated rainfall depths
            var w = new double[N];       // Importance sampling weights

            double sum = 0.0, sum2 = 0.0, sumWFinal = 0.0;

            // Perform importance-weighted simulation 
            for (int i = 0; i < N; i++)
            {
                // Sample storm centroid from proposal distribution
                double x = xISDist.InverseCDF(prngXY.NextDouble());
                double y = yISDist.InverseCDF(prngXY.NextDouble());

                // Sample storm shape (square side: 2, 4, or 6)
                int stormIndex = prngStorm.Next(0, 3);

                // Simulate depth based on overlap and storm randomness
                depths[i] = SST_Toy_Nonparametric_Simulator(x, y, stormIndex);

                // Compute importance sampling weight: p(x,y) / q(x,y)
                double p = xDist.PDF(x) * yDist.PDF(y);       // Uniform PDF
                double q = xISDist.PDF(x) * yISDist.PDF(y);   // Proposal PDF
                w[i] = (q > 0.0) ? p / q : 0.0;
                sumWFinal += w[i];

                // Apply weight to response
                double f = depths[i] * w[i];

                sum += f;
                sum2 += f * f;
            }

            // Estimate mean and standard error of basin-average rainfall
            double avg = sum / N;
            double avg2 = sum2 / N;
            double result = avg * domainArea;
            double standardError = Math.Sqrt((avg2 - avg * avg) / N) * domainArea;

            // Expected Output (N=10,000): Mean ≈ 1883.32, SE ≈ 51.54 (depends on overlap fraction and depth distribution)

            // Normalize weights for frequency curve plotting
            for (int i = 0; i < N; i++)
                w[i] /= sumWFinal;

            // Sort depths and weights together
            Array.Sort(depths, w); // Sorts depths and keeps weights aligned

            // Compute weighted plotting positions (cumulative sum of normalized weights)
            var pp = new double[N];
            pp[0] = w[0];
            for (int i = 1; i < N; i++)
                pp[i] = pp[i - 1] + w[i];

            // Print results for plotting depth-frequency curve
            for (int i = 0; i < N; i++)
            {
                Debug.WriteLine(depths[i] + "," + pp[i]);
            }

            // This frequency curve can now be plotted as:
            // x-axis: Return Period = 1 / (1 - pp)
            // y-axis: Depth
        }

        #endregion

    }
}

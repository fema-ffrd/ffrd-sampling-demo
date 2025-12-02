using Numerics;
using Numerics.Data;
using Numerics.Data.Statistics;
using Numerics.Distributions;
using Numerics.Sampling;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test_FFRD
{
    /// <summary>
    /// Test Monte Carlo simulation algorithms for sampling dam and levee breaches.
    /// Validates implementation against R-S formulation as documented in:
    /// "Investigations into Breach Sampling Techniques for FFRD" (Smith, Margo, Gonia, 2025)
    /// </summary>
    /// <remarks>
    /// <para>
    ///     <b> Authors: </b>
    ///     Haden Smith, USACE Risk Management Center, cole.h.smith@usace.army.mil
    /// </para>
    /// <para>
    ///     Each test method corresponds to an algorithm documented in the technical report
    ///     and validates results against RMC-TotalRisk benchmark within 1% relative tolerance
    ///     (5% for Value-at-Risk estimates due to higher variance in tail statistics).
    /// </para>
    /// </remarks>
    [TestClass]
    public class Test_BreachSampling
    {

        /// <summary>
        /// Algorithm 1: Verifies that Monte Carlo simulation using the annual maximum frequency 
        /// distribution accurately replicates results from the R-S formulation.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     This test establishes the baseline by sampling one hazard and one capacity per year
        ///     from the annual maximum distributions. Results are validated against RMC-TotalRisk
        ///     numerical integration within 1% relative tolerance.
        /// </para>
        /// <para>
        ///     <b>Key Parameters:</b>
        ///     - Hazard: GEV(μ=70, σ=15, ξ=0.05) 
        ///     - Capacity: Normal(μ=140, σ=30)
        ///     - Simulations: 1,000,000 years
        /// </para>
        /// <para>
        ///     <b>Expected Results:</b>
        ///     - APF ≈ 0.0522
        ///     - E[C_F] ≈ 2.99
        ///     - E[C_NF] ≈ 0.77
        ///     - E[C_T] ≈ 3.76
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm1_AnnualMaximum_Baseline()
        {
            // Define distributions:
            var Fs = new GeneralizedExtremeValue(70, 15, -0.05); // Hazard distribution (annual maximum)
            var Fr = new Normal(140, 30); // Capacity distribution

            // Define consequence functions (tabular, piecewise linear interpolation):
            var Cf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 }); // Consequences of failure
            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 }); // Consequences of non-failure

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed for reproducibility)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample annual maximum hazard level
                var s = Fs.InverseCDF(rng.NextDouble());

                // Sample resistance (capacity) level for this year
                var r = Fr.InverseCDF(rng.NextDouble());

                if (s > r)
                {
                    // Failure occurs when load exceeds resistance
                    pf += 1;
                    cf += Cf.Interpolate(s); // Accumulate failure consequences
                }
                else
                {
                    // No failure
                    cnf += Cnf.Interpolate(s); // Accumulate non-failure consequences
                }
            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])

            // Validate results against RMC-TotalRisk outputs (within 1% relative difference):
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
        }


        /// <summary>
        /// Algorithm 2: Verifies that Monte Carlo simulation using a Poisson point process
        /// with ONE capacity level per year accurately replicates R-S formulation results.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     This is the CORRECT approach for sampling capacity in a Poisson process framework.
        ///     A single capacity (resistance) value is drawn once per year and applied to ALL
        ///     flood events within that year. This maintains the intra-year correlation structure
        ///     and produces results consistent with the R-S formulation.
        /// </para>
        /// <para>
        ///     <b>Key Insight:</b> If the levee is particularly strong in a given year, all floods
        ///     within that year must overcome the same resistance. Sampling per event (Algorithm 3)
        ///     breaks this correlation and artificially inflates failure probability.
        /// </para>
        /// <para>
        ///     <b>Technical Details:</b>
        ///     - Converts GEV annual maximum to GPA conditional distribution using Madsen method
        ///     - Samples Poisson(λ=10) events per year
        ///     - Tracks maximum consequences per year (failure or non-failure)
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm2_PoissonProcess_OneCapacityPerYear_Correct()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method (Equations 15-17):
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // Hazard distribution (GPA)
            var Fr = new Normal(140, 30);                           // Capacity distribution
            var pois = new Poisson(lambda);                         // Poisson distribution for event count

            // Define consequence functions (tabular, piecewise linear interpolation):
            var Cf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 }); // Consequences of failure
            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 }); // Consequences of non-failure

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // CRITICAL: Sample a single capacity value for the ENTIRE year
                var r = Fr.InverseCDF(rng.NextDouble());

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level for this event
                    var s = Fs.InverseCDF(rng.NextDouble());

                    if (s > r)
                    {
                        // Failure occurs when load exceeds the year's capacity
                        failed = true;
                        mcf = Math.Max(mcf, Cf.Interpolate(s)); // Record maximum failure consequence
                    }
                    else
                    {
                        // No failure for this event
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate annual maximum failure consequences
                    // Note: No non-failure consequence is added in failure years (joint risk model)
                }
                else
                {
                    cnf += mcnf; // Accumulate annual maximum non-failure consequences
                }

            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])

            // Validate results against RMC-TotalRisk outputs (within 1% relative difference):
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
        }


        /// <summary>
        /// Algorithm 3: Demonstrates that sampling one capacity level per EVENT (INCORRECT approach)
        /// results in overestimation of APF and EAC compared to the R-S formulation.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     <b>WARNING: This method is intentionally WRONG to illustrate a common pitfall.</b>
        /// </para>
        /// <para>
        ///     When capacity is sampled independently for each flood event within a year, the intra-year
        ///     correlation is broken. Each event gets a new "chance" to find a weak capacity, artificially
        ///     inflating the annual failure probability. This is equivalent to computing:
        ///     P(any event fails) = 1 - ∏(1 - pⱼ) > P(S_max > R_year)
        /// </para>
        /// <para>
        ///     <b>Test Purpose:</b> The assertions use baseline RMC-TotalRisk results (which assume
        ///     one capacity per year). These assertions will FAIL, demonstrating the overestimation
        ///     effect documented in Table 2 of the report.
        /// </para>
        /// <para>
        ///     <b>DO NOT USE THIS APPROACH IN PRODUCTION CODE.</b>
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm3_PoissonProcess_OneCapacityPerEvent_INCORRECT_Demonstrates_Overestimation()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method:
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // Hazard distribution (GPA)
            var Fr = new Normal(140, 30);                           // Capacity distribution
            var pois = new Poisson(lambda);                         // Poisson distribution for event count

            // Define consequence functions (tabular, piecewise linear interpolation):
            var Cf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 }); // Consequences of failure
            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 }); // Consequences of non-failure

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());

                    // INCORRECT: Sample a NEW capacity value for each event (introduces bias)
                    var r = Fr.InverseCDF(rng.NextDouble());

                    if (s > r)
                    {
                        // Failure occurs (but with artificially inflated probability)
                        failed = true;
                        mcf = Math.Max(mcf, Cf.Interpolate(s)); // Record maximum failure consequence
                    }
                    else
                    {
                        // No failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                }
                else
                {
                    cnf += mcnf; // Accumulate non-failure consequences
                }

            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])

            // These assertions will FAIL because per-event sampling overestimates risk:
            // Expected: APF ≈ 0.0522, Actual: APF ≈ 0.0711 (36% overestimation)
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1% (expected failure - demonstrates overestimation)");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 1E-2, "E[C_F] differs by more than 1% (expected failure)");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1% (expected failure)");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 1E-2, "E[C_T] differs by more than 1% (expected failure)");
        }

        /// <summary>
        /// Algorithm 4: Verifies that Monte Carlo simulation using a step-function system response
        /// (representing SQRA critical load method) accurately replicates R-S formulation results.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     This test implements the correct approach for incorporating SQRA-derived failure modes
        ///     into a fully probabilistic Monte Carlo framework. The system response is modeled as a
        ///     step function: F_R|S(s) = {0 for s &lt; s_c; P_R|s≥s_c for s ≥ s_c}.
        /// </para>
        /// <para>
        ///     <b>Key Implementation:</b>
        ///     - One Bernoulli trial (r) is sampled per YEAR (not per event)
        ///     - Failure occurs if: (1) hazard exceeds critical load s_c, AND (2) r &lt; P_R|s≥s_c
        ///     - This correctly represents the conditional probability of failure given exceedance
        /// </para>
        /// <para>
        ///     <b>Verification Setup:</b>
        ///     - Critical load: s_c = 147.583 (0.01 AEP level)
        ///     - Conditional failure probability: P_R|s≥s_c = 0.5
        ///     - Back-calculated APF = 0.01 × 0.5 = 0.005
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm4_PoissonProcess_SQRA_StepFunction()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method:
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // Hazard distribution (GPA)

            // SQRA step-function parameters:
            double sc = 147.583; // Critical load threshold (0.01 AEP hazard level where failure mode activates)
            double PrSc = 0.5;   // Conditional probability of failure given hazard exceeds critical load

            var pois = new Poisson(lambda);                         // Poisson distribution for event count

            // Define consequence functions (tabular, piecewise linear interpolation):
            var Cf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 }); // Consequences of failure
            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 }); // Consequences of non-failure

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // CRITICAL: Sample ONE Bernoulli trial per YEAR (not per event)
                // This represents the annual capacity state for the step-function fragility
                var r = rng.NextDouble();

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level for this event
                    var s = Fs.InverseCDF(rng.NextDouble());

                    // Evaluate the step-function fragility:
                    // Failure occurs if BOTH conditions are met:
                    //   (1) Hazard exceeds critical load: s ≥ s_c
                    //   (2) Bernoulli trial succeeds: r < P_R|s≥s_c
                    if (s >= sc && r < PrSc)
                    {
                        // Failure occurs
                        failed = true;
                        mcf = Math.Max(mcf, Cf.Interpolate(s)); // Record maximum failure consequence
                    }
                    else
                    {
                        // No failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                }
                else
                {
                    cnf += mcnf; // Accumulate non-failure consequences
                }

            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])

            // Validate results against RMC-TotalRisk outputs (within 1% relative difference):
            Assert.AreEqual(0.00499982278063393, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(1.2577662504131, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(1.11443254342059, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(2.37219879383369, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
        }

        /// <summary>
        /// Algorithm 5: Verifies that Monte Carlo simulation with multiple competing failure modes
        /// accurately replicates R-S formulation results using the "weakest link" approach.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     This test implements competing risks analysis where multiple failure modes (PFMs)
        ///     can cause system failure. The governing mode is determined by finding the minimum
        ///     (weakest) capacity across all modes for each year.
        /// </para>
        /// <para>
        ///     <b>Key Features:</b>
        ///     - Five failure modes with different capacity distributions
        ///     - Each mode has its own unique consequence function
        ///     - One capacity value sampled per mode per year (maintains independence)
        ///     - First mode to fail governs the consequences
        /// </para>
        /// <para>
        ///     <b>Mathematical Basis:</b>
        ///     For independent failure modes, the system CDF is:
        ///     F_sys(s) = 1 - ∏[1 - F_i(s)]
        ///     
        ///     Sampling the minimum capacity is equivalent to sampling from this system CDF.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm5_PoissonProcess_MultipleCompetingFailureModes()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method:
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // Hazard distribution (GPA)

            // Define capacity distributions for the 5 potential failure modes (PFMs):
            var Fr1 = new Normal(140, 30);   // PFM 1: Overtopping
            var Fr2 = new Normal(160, 10);   // PFM 2: Internal erosion
            var Fr3 = new Normal(150, 20);   // PFM 3: Piping
            var Fr4 = new Normal(130, 35);   // PFM 4: Seepage
            var Fr5 = new Normal(160, 15);   // PFM 5: Slope instability

            // List of all failure mode distributions for easy iteration:
            var PFMs = new List<IUnivariateDistribution> { Fr1, Fr2, Fr3, Fr4, Fr5 };

            var pois = new Poisson(lambda);  // Poisson distribution for event count

            // Define mode-specific consequence functions:
            // Each failure mode has different breach characteristics and consequences
            var Cf1 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 });      // PFM 1 consequences
            var Cf2 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 3, 30, 300, 450, 600 });       // PFM 2 consequences
            var Cf3 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 10, 100, 1000, 1500, 2000 });  // PFM 3 consequences
            var Cf4 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 2, 20, 200, 300, 400 });       // PFM 4 consequences
            var Cf5 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 8, 80, 800, 1200, 1600 });     // PFM 5 consequences
            var Cf = new List<Linear>() { Cf1, Cf2, Cf3, Cf4, Cf5 };

            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 }); // Consequences of non-failure

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // Sample one capacity value per failure mode (represents annual 'resistance'):
                var r = new double[PFMs.Count];
                for (int j = 0; j < PFMs.Count; j++)
                {
                    r[j] = PFMs[j].InverseCDF(rng.NextDouble());
                }

                // Identify the governing (weakest) failure mode for this year:
                // The "weakest link" in a series system determines system failure
                double rMin = Tools.Min(r);         // Minimum capacity across all modes
                int rIdx = Tools.ArgMin(r);         // Index of the governing failure mode

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level for this event
                    var s = Fs.InverseCDF(rng.NextDouble());

                    if (s > rMin)
                    {
                        // Failure occurs when load exceeds the weakest mode's capacity
                        failed = true;
                        // Use consequence function of the governing failure mode
                        mcf = Math.Max(mcf, Cf[rIdx].Interpolate(s));
                    }
                    else
                    {
                        // No failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s));
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                }
                else
                {
                    cnf += mcnf; // Accumulate non-failure consequences
                }

            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])

            // Validate results against RMC-TotalRisk outputs (within 1% relative difference):
            Assert.AreEqual(0.142774399765792, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(3.53926379635317, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.48517856671668, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(4.02444236306986, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
        }

        /// <summary>
        /// Algorithm 6: Verifies that Monte Carlo simulation with multiple breach locations per levee segment
        /// accurately maintains segment-level APF through downscaling of conditional failure probability.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     This test addresses the "length effect" challenge: when multiple breach locations are simulated,
        ///     care must be taken to avoid artificially inflating the segment-level APF. The solution is to:
        ///     1. Determine the weakest failure mode at each breach location
        ///     2. Randomly select ONE breach location per year for evaluation
        ///     3. Apply downscaling: p_DS = 1 - (1 - p_site)^(1/N_sites)
        /// </para>
        /// <para>
        ///     <b>Physical Interpretation:</b>
        ///     The downscaling ensures that when aggregated over many simulations, the overall segment APF
        ///     matches the LST segment-level estimate. This prevents double-counting of failure when
        ///     adding spatial resolution.
        /// </para>
        /// <para>
        ///     <b>Implementation Note:</b>
        ///     The downscaling is applied as a secondary Bernoulli trial AFTER the capacity comparison.
        ///     This maintains the correct statistical structure while allowing spatial variability.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm6_PoissonProcess_MultipleBreachLocations_Method1()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method:
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // Hazard distribution (GPA)

            // Define capacity distributions for the 5 potential failure modes (PFMs):
            var Fr1 = new Normal(140, 30);   // PFM 1
            var Fr2 = new Normal(160, 10);   // PFM 2
            var Fr3 = new Normal(150, 20);   // PFM 3
            var Fr4 = new Normal(130, 35);   // PFM 4
            var Fr5 = new Normal(160, 15);   // PFM 5

            // List of all failure mode distributions:
            var PFMs = new List<IUnivariateDistribution> { Fr1, Fr2, Fr3, Fr4, Fr5 };

            var pois = new Poisson(lambda);  // Poisson distribution for event count

            // Define mode-specific consequence functions:
            var Cf1 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 });
            var Cf2 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 3, 30, 300, 450, 600 });
            var Cf3 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 10, 100, 1000, 1500, 2000 });
            var Cf4 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 2, 20, 200, 300, 400 });
            var Cf5 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 8, 80, 800, 1200, 1600 });
            var Cf = new List<Linear>() { Cf1, Cf2, Cf3, Cf4, Cf5 };

            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 });

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);               // Random number generator (fixed seed)
            int Nsites = 5;                                     // Number of potential breach locations

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // For each breach site, determine the weakest (minimum resistance) failure mode
                var rMin = new double[Nsites];    // Stores the weakest capacity per site
                var rIdx = new int[Nsites];       // Stores the index of the governing failure mode per site

                for (int k = 0; k < Nsites; ++k)
                {
                    double minR = double.MaxValue;
                    int minIdx = 0;

                    // Sample all failure modes at this breach location
                    for (int m = 0; m < PFMs.Count; m++)
                    {
                        var rm = PFMs[m].InverseCDF(rng.NextDouble());
                        if (rm < minR)
                        {
                            minR = rm;
                            minIdx = m;
                        }
                    }
                    rMin[k] = minR;
                    rIdx[k] = minIdx;
                }

                // METHOD 1: Randomly select ONE breach location for evaluation this year
                // This prevents over-counting when combined with downscaling
                int siteIdx = rng.Next(Nsites);
                double r = rMin[siteIdx];          // Selected site's weakest resistance
                int modeIdx = rIdx[siteIdx];       // Selected site's governing mode

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level for this event
                    var s = Fs.InverseCDF(rng.NextDouble());

                    // Evaluate if stage exceeds selected site's capacity
                    double breachIndicator = (s > r) ? 1.0 : 0.0;

                    // Apply downscaling to conditional failure probability (Equation 34)
                    // This ensures the system-wide APF matches the segment-level APF
                    // by accounting for spatial distribution across N_sites locations
                    double pDS = 1.0 - Math.Pow(1.0 - breachIndicator, 1.0 / Nsites);

                    // Perform Bernoulli trial with downscaled probability
                    if (rng.NextDouble() < pDS)
                    {
                        // Failure occurs
                        failed = true;
                        // Use consequence function of the governing failure mode at selected site
                        mcf = Math.Max(mcf, Cf[modeIdx].Interpolate(s));
                    }
                    else
                    {
                        // No failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s));
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                }
                else
                {
                    cnf += mcnf; // Accumulate non-failure consequences
                }

            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])

            // Validate results against RMC-TotalRisk outputs (within 1% relative difference):
            Assert.AreEqual(0.142774399765792, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(3.53926379635317, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.48517856671668, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(4.02444236306986, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
        }


        /// <summary>
        /// Algorithm 7: Verifies the HMS peak proxy method where breach timing is constrained to occur
        /// at or after the estimated peak from HEC-HMS hydrograph routing.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     This method approximates the R-S assumption (breach at peak) by using HEC-HMS output
        ///     as a proxy for peak timing. The approach:
        ///     1. Routes flood hydrograph with sampled peak scaled to hazard level
        ///     2. Samples peak timing from uncertainty distribution (Normal around HMS peak)
        ///     3. Allows breach only at or after estimated peak time
        /// </para>
        /// <para>
        ///     <b>Results:</b>
        ///     - APF matches R-S formulation (within 1%)
        ///     - EAC and VaR underestimate R-S due to timing errors (expected behavior)
        ///     - VaR validated within 5% tolerance due to higher variance
        /// </para>
        /// <para>
        ///     <b>Post-Breach Optimization:</b>
        ///     Once breach occurs, the event loop breaks immediately. Post-breach stages do not
        ///     contribute to non-failure consequences, which is physically realistic (area is
        ///     already inundated) and computationally efficient.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm7_PoissonProcess_HMSPeakProxy_BreachTiming()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method:
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // Hazard distribution (GPA)
            var Fr = new Normal(140, 30);                           // Capacity distribution
            var pois = new Poisson(lambda);                         // Poisson distribution for event count

            // Define consequence functions:
            var Cf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 });
            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 });

            // Define the hydrograph template (11 time steps, peaks at index 4):
            // Represents typical flood hydrograph shape: [52%, 62%, 73%, 85%, 100%, 90%, 81%, 73%, 66%, 59%, 53%]
            var hydro = new double[] { 104, 123, 145, 170, 200, 180, 162, 146, 131, 118, 106 };

            // Peak timing uncertainty distribution (models HMS vs RAS discrepancy)
            var timing = new Normal(4, 0.2);  // Mean at true peak (index 4), small std deviation

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences
            var cfVals = new double[N]; // Track all failure consequences for VaR calculation

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // Sample a single capacity value for the entire year
                var r = Fr.InverseCDF(rng.NextDouble());

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());
                    var scale = s / hydro.Max(); // Scale hydrograph to match sampled peak

                    // Sample timing of the peak from HMS proxy distribution
                    var time = Math.Floor(timing.InverseCDF(rng.NextDouble()));

                    // Route the scaled hydrograph
                    for (int k = 0; k < hydro.Length; k++)
                    {
                        var sh = hydro[k] * scale;

                        // Allow breach only at or after estimated peak time
                        if (sh > r && k >= time)
                        {
                            // Failure occurs
                            failed = true;
                            mcf = Math.Max(mcf, Cf.Interpolate(sh));
                            break; // OPTIMIZATION: Exit immediately after breach
                        }
                        else
                        {
                            // No failure yet
                            mcnf = Math.Max(mcnf, Cnf.Interpolate(sh));
                        }
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf;
                    cfVals[i] = mcf;
                }
                else
                {
                    cnf += mcnf;
                }
            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])
            double VaR = Statistics.Percentile(cfVals, 0.99); // 0.01 AEP Value-at-Risk

            // Validate results:
            // Note: EAC and VaR expected to be lower than R-S due to timing approximations
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 0.20, "E[C_F] differs by more than 20% (expected due to timing)");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 0.20, "E[C_T] differs by more than 20% (expected due to timing)");
            Assert.AreEqual(60.1708491184236, VaR, VaR * 0.25, "VaR differs by more than 25% (expected due to timing)");
        }

        /// <summary>
        /// Algorithm 8: Verifies the single-draw capacity method where a single failure threshold
        /// is sampled per year and breach occurs on first exceedance during the hydrograph.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     This is the RECOMMENDED method for operational FFRD implementation because:
        ///     1. Computationally efficient (no peak timing required a priori)
        ///     2. Mathematically equivalent to R-S for APF: P(S_max > R)
        ///     3. Supported directly by HEC-RAS breach triggers
        /// </para>
        /// <para>
        ///     <b>Method:</b>
        ///     - Sample one capacity R per year from marginal distribution F_R(r)
        ///     - Monitor entire hydrograph for first exceedance of R
        ///     - Trigger breach immediately when s(t) > R
        /// </para>
        /// <para>
        ///     <b>Trade-off:</b>
        ///     APF matches R-S formulation exactly, but EAC and VaR may be lower because
        ///     breach can occur before the hydrograph peak, potentially missing higher
        ///     consequences at the true peak stage.
        /// </para>
        /// <para>
        ///     <b>Physical Justification:</b>
        ///     Historical levee failures do not necessarily occur at peak stage. They can
        ///     occur on the rising limb due to cumulative effects. This method captures
        ///     that physical reality.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm8_PoissonProcess_SingleDraw_FirstPassage()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method:
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // Hazard distribution (GPA)
            var Fr = new Normal(140, 30);                           // Capacity distribution
            var pois = new Poisson(lambda);                         // Poisson distribution for event count

            // Define consequence functions:
            var Cf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 });
            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 });

            // Define the hydrograph template (11 time steps, peaks at 200):
            // Starting lower than Algorithm 7 to demonstrate early breach potential
            var hydro = new double[] { 48, 69, 98, 140, 200, 180, 162, 146, 131, 118, 106 };

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences
            var cfVals = new double[N]; // Track all failure consequences for VaR calculation

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // CRITICAL: Sample a single capacity threshold for the entire year
                // This is sampled from the MARGINAL distribution F_R(r), not conditional
                var r = Fr.InverseCDF(rng.NextDouble());

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());
                    var scale = s / hydro.Max(); // Scale hydrograph to match sampled peak

                    // Route the hydrograph - breach on FIRST exceedance
                    for (int k = 0; k < hydro.Length; k++)
                    {
                        var sh = hydro[k] * scale;

                        if (sh > r)
                        {
                            // Failure occurs on first exceedance (may be before peak)
                            failed = true;
                            mcf = Math.Max(mcf, Cf.Interpolate(sh));
                            break; // OPTIMIZATION: Exit immediately after breach
                        }
                        else
                        {
                            // No failure yet
                            mcnf = Math.Max(mcnf, Cnf.Interpolate(sh));
                        }
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf;
                    cfVals[i] = mcf;
                }
                else
                {
                    cnf += mcnf;
                }
            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])
            double VaR = Statistics.Percentile(cfVals, 0.99); // 0.01 AEP Value-at-Risk

            // Validate results:
            // Note: APF should match R-S exactly; EAC and VaR expected to be lower
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 0.30, "E[C_F] differs by more than 30% (expected due to early breach)");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 0.30, "E[C_T] differs by more than 30% (expected due to early breach)");
            Assert.AreEqual(60.1708491184236, VaR, VaR * 0.30, "VaR differs by more than 30% (expected due to early breach)");
        }

        /// <summary>
        /// Algorithm 8 (Alternative): Verifies the time-dependent first-passage method using
        /// cumulative hazard integration with high-water-mark optimization.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     This advanced method transforms the fragility curve into a time-dependent hazard function
        ///     λ(s) = f_R(s) / (1 - F_R(s)) and computes cumulative hazard H = -log(1 - F_R(s_max)).
        /// </para>
        /// <para>
        ///     <b>Optimization:</b>
        ///     The cumulative hazard H is only updated when the hydrograph reaches new high-water marks.
        ///     Since F_R(s) is monotonically increasing, this is mathematically equivalent to the
        ///     full integral but computationally more efficient.
        /// </para>
        /// <para>
        ///     <b>Equivalence to Single-Draw:</b>
        ///     For monotonically rising (or single-peaked) hydrographs:
        ///     p_F = 1 - exp(-∫λ(s(t))dt) = F_R(S_max)
        ///     
        ///     This is mathematically identical to the single-draw capacity method (Algorithm 8).
        /// </para>
        /// <para>
        ///     <b>Practical Note:</b>
        ///     While theoretically elegant, this method requires a priori knowledge of the entire
        ///     hydrograph to compute the integral, limiting its feasibility in dynamic simulations.
        ///     The single-draw method (Algorithm 8) is recommended for operational use.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm8_Alternative_TimeDependent_FirstPassage_CumulativeHazard()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method:
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // Hazard distribution (GPA)
            var Fr = new Normal(140, 30);                           // Capacity distribution
            var pois = new Poisson(lambda);                         // Poisson distribution for event count

            // Define consequence functions:
            var Cf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 });
            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 });

            // Define the hydrograph template:
            var hydro = new double[] { 48, 69, 98, 140, 200, 180, 162, 146, 131, 118, 106 };

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences
            var cfVals = new double[N]; // Track all failure consequences for VaR calculation

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // Sample a failure threshold from Exp(1) distribution
                // This represents: when does cumulative hazard exceed threshold?
                double rThresh = -Math.Log(1 - rng.NextDouble());

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());
                    var scale = s / hydro.Max(); // Scale hydrograph to match sampled peak

                    double H = 0;           // Cumulative hazard
                    double maxStage = 0;    // Track high-water mark

                    // Route hydrograph and integrate hazard
                    for (int k = 0; k < hydro.Length; k++)
                    {
                        double sh = hydro[k] * scale;

                        // OPTIMIZATION: Only update hazard at new high-water marks
                        // This is equivalent to full integral since λ(s) is monotonic in s
                        if (sh > maxStage)
                        {
                            maxStage = sh;
                            // Exact hazard integral up to this stage: H = ∫λ(s)ds = -log(1 - F_R(s))
                            H = -Math.Log(1.0 - Fr.CDF(sh));
                        }

                        if (H >= rThresh) // First passage: cumulative hazard exceeds threshold
                        {
                            // Failure occurs
                            failed = true;
                            mcf = Math.Max(mcf, Cf.Interpolate(sh));
                            break; // OPTIMIZATION: Exit immediately after breach
                        }
                        else
                        {
                            // No failure yet
                            mcnf = Math.Max(mcnf, Cnf.Interpolate(sh));
                        }
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf;
                    cfVals[i] = mcf;
                }
                else
                {
                    cnf += mcnf;
                }
            }

            // Post-process simulation results:
            double apf = pf / N;       // Annual probability of failure (APF)
            double eCf = cf / N;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / N;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])
            double VaR = Statistics.Percentile(cfVals, 0.99); // 0.01 AEP Value-at-Risk

            // Validate results:
            // Should match single-draw method (Algorithm 8) exactly
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 0.30, "E[C_F] differs by more than 30% (expected due to early breach)");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 0.30, "E[C_T] differs by more than 30% (expected due to early breach)");
            Assert.AreEqual(60.1708491184236, VaR, VaR * 0.30, "VaR differs by more than 30% (expected due to early breach)");
        }


        /// <summary>
        /// Algorithm 9: Verifies that importance sampling maintains unbiased risk estimates
        /// when properly weighted using likelihood ratios.
        /// </summary>
        /// <remarks>
        /// <para>
        ///     Importance sampling improves computational efficiency by oversampling rare (high-impact)
        ///     flood events from a proposal distribution q(s), then reweighting results by w = p(s)/q(s).
        /// </para>
        /// <para>
        ///     <b>Critical Implementation Details:</b>
        ///     - Weights are accumulated in LOG-SPACE to prevent numerical underflow
        ///     - Annual weight = product of all event weights within that year
        ///     - Self-normalized estimator: divide weighted sums by total weight sum
        /// </para>
        /// <para>
        ///     <b>Proposal Distribution Design:</b>
        ///     The proposal uses slightly heavier tails (σ increased by 20%, ξ = -0.1) to increase
        ///     sampling of extreme events. Care must be taken not to over-focus on extremes, as this
        ///     can lead to high variance in importance weights.
        /// </para>
        /// <para>
        ///     <b>Practical Guidance:</b>
        ///     For FFRD/SST applications, importance sampling should be used selectively on large
        ///     watersheds where rare events dominate risk but are computationally expensive to sample.
        ///     Always validate that weighted estimates match crude Monte Carlo results before
        ///     relying on importance sampling for operational decisions.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Algorithm9_PoissonProcess_ImportanceSampling_LogSpace()
        {
            // Define parameters for the Poisson process and hazard distribution:
            double lambda = 10;                      // Average number of flood events per year
            double locGEV = 70;                      // GEV location parameter
            double sclGEV = 15;                      // GEV scale parameter
            double shp = 0.05;                       // GEV shape parameter

            // Convert GEV parameters to GPA using the Madsen method:
            double locGPA = locGEV - sclGEV / shp * (1 - Math.Pow(lambda, -shp));
            double sclGPA = sclGEV * Math.Pow(lambda, -shp);

            var Fs = new GeneralizedPareto(locGPA, sclGPA, -shp);  // TARGET distribution (GPA)
            var Fr = new Normal(140, 30);                           // Capacity distribution
            var pois = new Poisson(lambda);                         // Poisson distribution for event count

            // PROPOSAL distribution for importance sampling (heavier tail to oversample extremes)
            var FsIS = new GeneralizedPareto(locGPA, sclGPA * 1.2, -0.1);

            // Define consequence functions:
            var Cf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 });
            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 });

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Weighted counter for number of failures
            double cf = 0;   // Weighted sum of failure consequences
            double cnf = 0;  // Weighted sum of non-failure consequences
            double totalWeight = 0;     // Total sum of importance weights

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // Sample a single capacity value for the entire year
                var r = Fr.InverseCDF(rng.NextDouble());

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                // CRITICAL: Accumulate weights in LOG-SPACE to avoid numerical underflow
                // Annual weight = ∏ᵢ w(sᵢ) = exp(∑ᵢ log(w(sᵢ)))
                double logWeight = 0.0;

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level from PROPOSAL distribution
                    var s = FsIS.InverseCDF(rng.NextDouble());

                    // Compute importance weight in log-space:
                    // w = p(s)/q(s) → log(w) = log(p(s)) - log(q(s))
                    double logP = Fs.LogPDF(s);      // Log TARGET density
                    double logQ = FsIS.LogPDF(s);    // Log PROPOSAL density
                    logWeight += logP - logQ;        // Accumulate log-weight

                    if (s > r)
                    {
                        // Failure occurs
                        failed = true;
                        mcf = Math.Max(mcf, Cf.Interpolate(s));
                    }
                    else
                    {
                        // No failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s));
                    }
                }

                // Convert accumulated log-weight back to normal space
                double yearWeight = Math.Exp(logWeight);

                if (failed)
                {
                    pf += yearWeight;
                    cf += yearWeight * mcf;    // Weight the failure consequences
                }
                else
                {
                    cnf += yearWeight * mcnf; // Weight the non-failure consequences
                }

                totalWeight += yearWeight;
            }

            // Post-process simulation results using SELF-NORMALIZED estimator:
            // Dividing by total weight ensures unbiased estimates even if weights aren't perfectly normalized
            double apf = pf / totalWeight;       // Annual probability of failure (APF)
            double eCf = cf / totalWeight;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / totalWeight;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;             // Total expected annual consequences (E[C_T])

            // Validate results against RMC-TotalRisk outputs (within 1% relative difference):
            Assert.AreEqual(0.0522145425679118, apf, apf * 0.01, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 0.01, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 0.01, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 0.01, "E[C_T] differs by more than 1%");
        }

    }
}

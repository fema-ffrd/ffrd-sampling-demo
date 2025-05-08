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
    /// Test Monte Carlo simulation algorithms for sampling levee breaches.
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
    public class Test_LeveeBreaches
    {

        /// <summary>
        /// Verifies that the Monte Carlo simulation of annual maximum flood risk
        /// accurately replicates results from the R–S formulation as implemented in RMC-TotalRisk.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The test asserts that the Monte Carlo results are within 1% of these expected values.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_AnnualMaximumSimulation()
        {
            // Define distributions:
            var Fs = new GeneralizedExtremeValue(70, 15, -0.05); // Hazard distribution
            var Fr = new Normal(140, 30); // Capacity distribution

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
                // Sample annual peak hazard level
                var s = Fs.InverseCDF(rng.NextDouble());

                // Sample resistance (capacity) level
                var r = Fr.InverseCDF(rng.NextDouble());

                if (s > r)
                {
                    // failure
                    pf += 1;
                    cf += Cf.Interpolate(s); // Accumulate failure consequences
                }
                else
                {
                    // no failure
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
        /// Verifies that the Monte Carlo simulation of a Poisson point process
        /// with one capacity level per year accurately replicates the results
        /// from the R–S formulation as implemented in RMC-TotalRisk.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The test asserts that the Monte Carlo results are within 1% of these expected values.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerYear()
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

                // Sample a single capacity value for the entire year
                var r = Fr.InverseCDF(rng.NextDouble());

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());

                    if (s > r)
                    {
                        // failure
                        failed = true;
                        mcf = Math.Max (mcf, Cf.Interpolate(s)); // Record maximum failure consequence
                    }
                    else
                    {
                        // no failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                    // No non-failure consequence is added in failure years
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
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
        }


        /// <summary>
        /// Demonstrates that a Monte Carlo simulation of a Poisson point process
        /// with one capacity level per event (rather than per year) results in
        /// overestimation of the annual probability of failure (APF) and expected annual consequences (EAC).
        /// </summary>
        /// <remarks>
        /// <para>
        /// This test highlights a key pitfall: when simulating flood risk, sampling a new capacity (resistance)
        /// for each flood event within a year introduces a bias that inflates both the APF and EAC compared to
        /// the R–S formulation.
        /// </para>
        /// <para>
        /// The simulation uses a Poisson process (λ = 10) to determine the number of flood events per year,
        /// sampling hazards from a Generalized Pareto distribution and capacities from a Normal distribution.
        /// The failure and non-failure consequences are defined as piecewise linear functions.
        /// </para>
        /// <para>
        /// Important: The assertions use expected results from RMC-TotalRisk (which assume one capacity per year).
        /// Therefore, these assertions are expected to fail (or show deviation &gt; 1%), illustrating the overestimation effect.
        /// In practice, this test serves as a diagnostic to confirm the flaw in event-based capacity sampling.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerEvent()
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

                    // Sample a new capacity value for this event (note: introduces bias)
                    var r = Fr.InverseCDF(rng.NextDouble());

                    if (s > r)
                    {
                        // failure
                        failed = true;
                        mcf = Math.Max(mcf, Cf.Interpolate(s)); // Record maximum failure consequence
                    }
                    else
                    {
                        // no failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                    // No non-failure consequence is added in failure years
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

            // Validate results (note: expected to deviate due to overestimation effect):
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
        }

        /// <summary>
        /// Verifies that the Monte Carlo simulation of a Poisson point process
        /// with a step-function response (representing an SQRA-style critical load failure mode)
        /// accurately replicates results from the R–S formulation as implemented in RMC-TotalRisk.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The test simulates flood events over 1,000,000 years, where each year consists of a Poisson-distributed
        /// number of flood events. For each flood event, a peak hazard is sampled, and failure is determined
        /// using a step-function fragility curve: the system can fail only if the hazard exceeds the critical load
        /// and a Bernoulli trial (sampled once per year) indicates failure. This mirrors the behavior of a
        /// semi-quantitative risk assessment (SQRA) critical load method.
        /// </para>
        /// <para>
        /// The simulation results are compared to expected values (from RMC-TotalRisk) and asserted to be within 1%
        /// relative difference. The key objective is to confirm that the step-function approach produces
        /// risk metrics that align with the expected APF, failure consequences, non-failure consequences, and total risk.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerYear_SQRA_StepFunction()
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
            double sc = 147.583; // Critical load threshold (hazard level where failure mode activates)
            double PrSc = 0.5;   // Conditional probability of failure if hazard exceeds critical load
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

                // Sample one Bernoulli trial per year to determine if failure occurs (if critical load is exceeded)
                var r = rng.NextDouble();

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());

                    // Evaluate the step-function fragility
                    if (s >= sc && r < PrSc)
                    {
                        // failure
                        failed = true;
                        mcf = Math.Max(mcf, Cf.Interpolate(s)); // Record maximum failure consequence
                    }
                    else
                    {
                        // no failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                    // No non-failure consequence is added in failure years
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
        /// Verifies that the Monte Carlo simulation of a Poisson point process
        /// with multiple failure modes and one capacity level per year accurately
        /// replicates the results from the R–S formulation as implemented in RMC-TotalRisk.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This test simulates a levee segment with five failure modes, each having its own
        /// capacity distribution and consequence function. The framework ensures that the
        /// governing failure mode (the weakest link) is correctly identified each year,
        /// and that the corresponding consequences are applied. The test asserts that the
        /// Monte Carlo results are within 1% of the expected values computed using 
        /// numerical integration.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerYear_MultipleFailureModes()
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

            // List of all failure mode distributions for easy looping:
            var PFMs = new List<IUnivariateDistribution> { Fr1, Fr2, Fr3, Fr4, Fr5 };

            var pois = new Poisson(lambda);  // Poisson distribution for event count

            // Define consequence functions (tabular, piecewise linear interpolation):
            var Cf1 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 }); // Consequences of failure for PFM 1
            var Cf2 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 3, 30, 300, 450, 600 }); // Consequences of failure for PFM 2
            var Cf3 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 10, 100, 1000, 1500, 2000 }); // Consequences of failure for PFM 2
            var Cf4 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 2, 20, 200, 300, 400 }); // Consequences of failure for PFM 2
            var Cf5 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 8, 80, 800, 1200, 1600 }); // Consequences of failure for PFM 2
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

                // Sample one capacity value per failure mode (represents the annual 'resistance'):
                var r = new double[PFMs.Count];
                for (int j = 0; j < PFMs.Count; j++)
                {
                    r[j] = PFMs[j].InverseCDF(rng.NextDouble());
                }
                // Identify the governing (weakest) failure mode for this year:
                double rMin = Tools.Min(r);         // Minimum capacity
                int rIdx = Tools.ArgMin(r);         // Index of the governing failure mode

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());

                    if (s > rMin)
                    {
                        // failure
                        failed = true;
                        mcf = Math.Max(mcf, Cf[rIdx].Interpolate(s)); // Record maximum failure consequence
                    }
                    else
                    {
                        // no failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                    // No non-failure consequence is added in failure years
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
        /// Verifies that the Monte Carlo simulation of a Poisson point process 
        /// with multiple failure modes and multiple breach locations per levee segment 
        /// accurately replicates the results from the R–S formulation as implemented in RMC-TotalRisk.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This test extends the competing risks simulation by adding multiple breach locations 
        /// per levee segment (spatial variability). It ensures that even when multiple breach locations 
        /// are sampled, the overall segment APF and risk metrics are consistent with the segment-level 
        /// APF from LST 2.0 by appropriately rescaling the conditional probability of failure.
        /// </para>
        /// <para>
        /// The method uses a downscaling approach to adjust the conditional failure probability 
        /// at each site, ensuring the combined risk across all breach locations matches the 
        /// expected segment-level probability of failure.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerYear_MultipleFailureModes_MultipleBreachLocations()
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

            // List of all failure mode distributions for easy looping:
            var PFMs = new List<IUnivariateDistribution> { Fr1, Fr2, Fr3, Fr4, Fr5 };

            var pois = new Poisson(lambda);  // Poisson distribution for event count

            // Define consequence functions (tabular, piecewise linear interpolation):
            var Cf1 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 5, 50, 500, 750, 1000 }); // Consequences of failure for PFM 1
            var Cf2 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 3, 30, 300, 450, 600 }); // Consequences of failure for PFM 2
            var Cf3 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 10, 100, 1000, 1500, 2000 }); // Consequences of failure for PFM 2
            var Cf4 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 2, 20, 200, 300, 400 }); // Consequences of failure for PFM 2
            var Cf5 = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 8, 80, 800, 1200, 1600 }); // Consequences of failure for PFM 2
            var Cf = new List<Linear>() { Cf1, Cf2, Cf3, Cf4, Cf5 };

            var Cnf = new Linear(new double[] { 60, 100, 140, 200, 250, 350 }, new double[] { 0, 1, 10, 100, 150, 200 }); // Consequences of non-failure

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

                // Randomly select a breach location for evaluation this year
                int siteIdx = rng.Next(Nsites);
                double r = rMin[siteIdx];          // Selected site's weakest resistance

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());

                    // Downscale the failure probability to avoid over-counting across breach sites
                    // This ensures the system-wide APF remains consistent with LST 2.0 assumptions
                    double pDS = 1.0 - Math.Pow(1.0 - (s > r ? 1.0 : 0.0), 1.0 / Nsites);

                    if (rng.NextDouble() < pDS)
                    {
                        // failure
                        failed = true;
                        mcf = Math.Max(mcf, Cf[rIdx[siteIdx]].Interpolate(s)); // Record maximum failure consequence

                    }
                    else
                    {
                        // no failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                    // No non-failure consequence is added in failure years
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
        /// Verifies that the Monte Carlo simulation of a Poisson point process with a peak-timing proxy
        /// (e.g., from HEC-HMS) accurately reproduces the expected APF as implemented in RMC-TotalRisk. 
        /// However, the EAC and VaR are underestimated due to early or late breach errors.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method tests the HMS peak proxy approach by simulating flood hydrographs with slight timing uncertainties.
        /// For each event:
        /// - A flood hydrograph is scaled according to the sampled peak stage.
        /// - The timing of the true peak is sampled from a Normal distribution around the expected peak.
        /// - Failure is only allowed after the peak timing is reached.
        /// The test tracks failure probability, expected annual consequences (EAC), and 0.01 AEP Value-at-Risk (VaR),
        /// and compares them against reference results (from RMC-TotalRisk) within a tolerance range.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerYear_HMSPeakProxy()
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

            // Define the hydrograph template
            var hydro = new double[] { 104, 123, 145, 170, 200, 180, 162, 146, 131, 118, 106 };
            var timing = new Normal(4, 0.2);  // Peak timing uncertainty distribution (around index 4)

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)
            
            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences
            var cfVals = new double[N]; // keep track of all values to run summary stats

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
                    var scale = s / hydro.Max(); // hydrograph scale factor

                    // Sample timing of the peak from the proxy distribution (HMS error model)
                    var time = Math.Floor(timing.InverseCDF(rng.NextDouble()));

                    // Route the hydrograph (scaled)
                    for (int k = 0; k < hydro.Length; k++)
                    {
                        var sh = hydro[k] * scale;

                        if (sh > r && k >= time)
                        {
                            // failure
                            failed = true;
                            mcf = Math.Max(mcf, Cf.Interpolate(sh)); // Record maximum failure consequence
                            break;
                        }
                        else
                        {
                            // no failure
                            mcnf = Math.Max(mcnf, Cnf.Interpolate(sh)); // Record maximum non-failure consequence
                        }

                    }

                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                    cfVals[i] = mcf;
                    // No non-failure consequence is added in failure years
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
            double VaR = Statistics.Percentile(cfVals, 0.99); // 0.01 AEP Value-at-Risk

            // Validate results (note: expected to deviate due to underestimation effect):
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
            Assert.AreEqual(60.1708491184236, VaR, VaR * 0.05, "VaR differs by more than 5%");
        }

        /// <summary>
        /// Verifies that the Monte Carlo simulation using the single-draw capacity method
        /// accurately reproduces the expected APF as implemented in RMC-TotalRisk. 
        /// However, the EAC and VaR are underestimated due to early or late breach errors.
        /// </summary>
        /// <remarks>
        /// <para>
        /// In this test, a single failure threshold (capacity) is sampled once per year, and the flood hydrograph
        /// is monitored to see if any time-step exceeds the threshold. This method replicates p_F = P(S_max > R).
        /// The test tracks:
        /// - Annual probability of failure (APF)
        /// - Expected annual consequences (failure and non-failure)
        /// - 0.01 AEP Value-at-Risk (VaR)
        /// The results are compared against reference outputs from RMC-TotalRisk within a tolerance margin.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerYear_SingleDraw()
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

            // Define the hydrograph template
            var hydro = new double[] { 48, 69, 98, 140, 200, 180, 162, 146, 131, 118, 106 };

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences
            var cfVals = new double[N]; // keep track of all values to run summary stats

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
                    var scale = s / hydro.Max(); // hydrograph scale factor

                    // Route the hydrograph
                    for (int k = 0; k < hydro.Length; k++)
                    {
                        var sh = hydro[k] * scale;

                        if (sh > r)
                        {
                            // failure
                            failed = true;
                            mcf = Math.Max(mcf, Cf.Interpolate(sh)); // Record maximum failure consequence
                            break;
                        }
                        else
                        {
                            // no failure
                            mcnf = Math.Max(mcnf, Cnf.Interpolate(sh)); // Record maximum non-failure consequence
                        }

                    }

                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                    cfVals[i] = mcf;
                    // No non-failure consequence is added in failure years
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
            double VaR = Statistics.Percentile(cfVals, 0.99); // 0.01 AEP Value-at-Risk

            // Validate results (note: expected to deviate due to underestimation effect):
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
            Assert.AreEqual(60.1708491184236, VaR, VaR * 0.05, "VaR differs by more than 5%");
        }

        /// <summary>
        /// Verifies that the Monte Carlo simulation using the time-dependent first-passage method
        /// reproduces the expected APF as implemented in RMC-TotalRisk. 
        /// However, the EAC and VaR are underestimated due to early or late breach errors.
        /// </summary>
        /// <remarks>
        /// <para>
        /// In this test, the fragility curve is transformed into a time-dependent hazard function.
        /// The simulation integrates the hazard over the hydrograph, simulating a "first-passage" process:
        /// - Each hydrograph is routed, and the cumulative hazard H is computed dynamically as flood stages rise.
        /// - A threshold rThresh ~ Exp(1) is drawn per year, and failure is triggered the first time H exceeds rThresh.
        /// - This replicates the cumulative probability of failure: p_F = 1 - exp(-∫λ(s(t)) dt), where λ(s) is the hazard rate.
        /// The test tracks APF, expected annual consequences (failure/non-failure), and 0.01 AEP VaR, compared to benchmark results.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerYear_TimeDependent_FirstPassage()
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

            // Define the hydrograph template
            var hydro = new double[] { 48, 69, 98, 140, 200, 180, 162, 146, 131, 118, 106 };

            // Initialize simulation parameters:
            int N = 1000000;                                    // Number of years to simulate
            var rng = new MersenneTwister(12345);                // Random number generator (fixed seed)

            // Initialize accumulators:
            double pf = 0;   // Counter for number of failures
            double cf = 0;   // Sum of failure consequences
            double cnf = 0;  // Sum of non-failure consequences
            var cfVals = new double[N]; // keep track of all values to run summary stats

            // Monte Carlo simulation loop:
            for (int i = 0; i < N; i++)
            {
                // Sample the number of flood events in the current year
                var e = pois.InverseCDF(rng.NextDouble());

                // Sample one Bernoulli trial per year to determine if failure occurs
                double rThresh = -Math.Log(1 - rng.NextDouble());

                bool failed = false;
                double mcf = 0; // max of failure consequences
                double mcnf = 0; // max of non-failure consequences

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level
                    var s = Fs.InverseCDF(rng.NextDouble());
                    var scale = s / hydro.Max(); // hydrograph scale factor

                    double H = 0;
                    double maxStage = 0;

                    for (int k = 0; k < hydro.Length; k++)
                    {
                        double sh = hydro[k] * scale;

                        // only update when we hit a new high‐water mark:
                        if (sh > maxStage)
                        {
                            maxStage = sh;
                            // exact hazard integral up to this stage:
                            H = -Math.Log(1.0 - Fr.CDF(sh));
                        }

                        if (H >= rThresh) // first passage
                        {
                            // failure
                            failed = true;
                            mcf = Math.Max(mcf, Cf.Interpolate(sh)); // Record maximum failure consequence
                            break;
                        }
                        else
                        {
                            // no failure
                            mcnf = Math.Max(mcnf, Cnf.Interpolate(sh)); // Record maximum non-failure consequence
                        }

                    }

                }

                if (failed)
                {
                    pf += 1;
                    cf += mcf; // Accumulate failure consequences
                    cfVals[i] = mcf;
                    // No non-failure consequence is added in failure years
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
            double VaR = Statistics.Percentile(cfVals, 0.99); // 0.01 AEP Value-at-Risk

            // Validate results (note: expected to deviate due to underestimation effect):
            Assert.AreEqual(0.0522145425679118, apf, apf * 1E-2, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 1E-2, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 1E-2, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 1E-2, "E[C_T] differs by more than 1%");
            Assert.AreEqual(60.1708491184236, VaR, VaR * 0.05, "VaR differs by more than 5%");
        }


        /// <summary>
        /// Verifies that the Monte Carlo simulation of a Poisson point process 
        /// with importance sampling accurately reproduces the R–S formulation results.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This test implements importance sampling (IS) to improve the efficiency of 
        /// simulating rare flood events. The Poisson process samples the number of 
        /// flood events per year, and each flood event samples its peak stage from 
        /// an alternative ("proposal") distribution. Importance weights are calculated 
        /// as the likelihood ratio between the target (GPA) distribution and the proposal.
        /// </para>
        /// <para>
        /// The simulation aggregates annual-level weights by computing the product 
        /// of weights for all flood events within each year (accumulated in log-space 
        /// for numerical stability). A self-normalized estimator is used by dividing 
        /// accumulated weighted sums by the total weight, ensuring unbiased estimates.
        /// </para>
        /// <para>
        /// The results are validated against RMC-TotalRisk outputs within a 1% relative difference.
        /// </para>
        /// </remarks>
        [TestMethod]
        public void Test_PoissonProcess_OneCapacityLevelPerYear_ImportanceSampling()
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

            var FsIS = new GeneralizedPareto(locGPA, sclGPA * 1.2, -0.1); // Importance sampling distribution (GPA with heavier tail)

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
                double logWeight = 0.0; // Log-space accumulator to avoid numerical instability

                for (int j = 0; j < e; j++)
                {
                    // Sample peak hazard level from Importance distribution
                    var s = FsIS.InverseCDF(rng.NextDouble());

                    // Compute the importance sampling weight in log-space
                    double logP = Fs.LogPDF(s);      // Log target density
                    double logQ = FsIS.LogPDF(s);    // Log proposal density
                    logWeight += logP - logQ;        // Accumulate log-weight

                    if (s > r)
                    {
                        // failure
                        failed = true;
                        mcf = Math.Max(mcf, Cf.Interpolate(s)); // Record maximum failure consequence
                    }
                    else
                    {
                        // no failure
                        mcnf = Math.Max(mcnf, Cnf.Interpolate(s)); // Record maximum non-failure consequence
                    }
                }

                // Convert accumulated log-weight to normal space
                double yearWeight = Math.Exp(logWeight);

                if (failed)
                {
                    pf += yearWeight;
                    cf += yearWeight * mcf;    // Accumulate failure consequences
                }
                else
                {
                    cnf += yearWeight * mcnf; // Accumulate non-failure consequences
                }

                totalWeight += yearWeight;
            }

            // Post-process simulation results:
            double apf = pf / totalWeight;       // Annual probability of failure (APF)
            double eCf = cf / totalWeight;       // Expected annual failure consequences (E[C_F])
            double eCnf = cnf / totalWeight;     // Expected annual non-failure consequences (E[C_NF])
            double eCt = eCf + eCnf;   // Total expected annual consequences (E[C_T])

            // Validate results against RMC-TotalRisk outputs (within 1% relative difference):
            Assert.AreEqual(0.0522145425679118, apf, apf * 0.01, "APF differs by more than 1%");
            Assert.AreEqual(2.99043811042954, eCf, eCf * 0.01, "E[C_F] differs by more than 1%");
            Assert.AreEqual(0.767847326170599, eCnf, eCnf * 0.01, "E[C_NF] differs by more than 1%");
            Assert.AreEqual(3.75828543660014, eCt, eCt * 0.01, "E[C_T] differs by more than 1%");
        }

    }
}

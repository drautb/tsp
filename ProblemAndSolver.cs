using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Windows.Forms;
using System.Diagnostics;

namespace TSP
{

    class ProblemAndSolver
    {

        private class TSPSolution
        {
            /// <summary>
            /// we use the representation [cityB,cityA,cityC] 
            /// to mean that cityB is the first city in the solution, cityA is the second, cityC is the third 
            /// and the edge from cityC to cityB is the final edge in the path.  
            /// You are, of course, free to use a different representation if it would be more convenient or efficient 
            /// for your node data structure and search algorithm. 
            /// </summary>
            public ArrayList
                Route;

            public TSPSolution(ArrayList iroute)
            {
                Route = new ArrayList(iroute);
            }


            /// <summary>
            /// Compute the cost of the current route.  
            /// Note: This does not check that the route is complete.
            /// It assumes that the route passes from the last city back to the first city. 
            /// </summary>
            /// <returns></returns>
            public double costOfRoute()
            {
                // go through each edge in the route and add up the cost. 
                int x;
                City here;
                double cost = 0D;

                for (x = 0; x < Route.Count - 1; x++)
                {
                    here = Route[x] as City;
                    cost += here.costToGetTo(Route[x + 1] as City);
                }

                // go from the last city to the first. 
                here = Route[Route.Count - 1] as City;
                cost += here.costToGetTo(Route[0] as City);
                return cost;
            }
        }

        #region Private members 

        /// <summary>
        /// Default number of cities (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Problem Size text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int DEFAULT_SIZE = 25;

        private const int CITY_ICON_SIZE = 5;

        // For normal and hard modes:
        // hard mode only
        private const double FRACTION_OF_PATHS_TO_REMOVE = 0.20;

        /// <summary>
        /// the cities in the current problem.
        /// </summary>
        private City[] Cities;
        /// <summary>
        /// a route through the current problem, useful as a temporary variable. 
        /// </summary>
        private ArrayList Route;
        /// <summary>
        /// best solution so far. 
        /// </summary>
        private TSPSolution bssf; 

        /// <summary>
        /// how to color various things. 
        /// </summary>
        private Brush cityBrushStartStyle;
        private Brush cityBrushStyle;
        private Pen routePenStyle;


        /// <summary>
        /// keep track of the seed value so that the same sequence of problems can be 
        /// regenerated next time the generator is run. 
        /// </summary>
        private int _seed;
        /// <summary>
        /// number of cities to include in a problem. 
        /// </summary>
        private int _size;

        /// <summary>
        /// Difficulty level
        /// </summary>
        private HardMode.Modes _mode;

        /// <summary>
        /// random number generator. 
        /// </summary>
        private Random rnd;

        /// <summary>
        /// true random number generator. This generator is not influenced by the seed,
        /// thus is can return random results even when the same seed is used.
        /// </summary>
        private Random trueRand = new Random();
        #endregion

        #region Public members
        public int Size
        {
            get { return _size; }
        }

        public int Seed
        {
            get { return _seed; }
        }
        #endregion

        #region Constructors
        public ProblemAndSolver()
        {
            this._seed = 1; 
            rnd = new Random(1);
            this._size = DEFAULT_SIZE;

            this.resetData();
        }

        public ProblemAndSolver(int seed)
        {
            this._seed = seed;
            rnd = new Random(seed);
            this._size = DEFAULT_SIZE;

            this.resetData();
        }

        public ProblemAndSolver(int seed, int size)
        {
            this._seed = seed;
            this._size = size;
            rnd = new Random(seed); 
            this.resetData();
        }
        #endregion

        #region Private Methods

        /// <summary>
        /// Reset the problem instance.
        /// </summary>
        private void resetData()
        {

            Cities = new City[_size];
            Route = new ArrayList(_size);
            bssf = null;

            if (_mode == HardMode.Modes.Easy)
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());
            }
            else // Medium and hard
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() * City.MAX_ELEVATION);
            }

            HardMode mm = new HardMode(this._mode, this.rnd, Cities);
            if (_mode == HardMode.Modes.Hard)
            {
                int edgesToRemove = (int)(_size * FRACTION_OF_PATHS_TO_REMOVE);
                mm.removePaths(edgesToRemove);
            }
            City.setModeManager(mm);

            cityBrushStyle = new SolidBrush(Color.Black);
            cityBrushStartStyle = new SolidBrush(Color.Red);
            routePenStyle = new Pen(Color.Blue,1);
            routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// make a new problem with the given size.
        /// </summary>
        /// <param name="size">number of cities</param>
        //public void GenerateProblem(int size) // unused
        //{
        //   this.GenerateProblem(size, Modes.Normal);
        //}

        /// <summary>
        /// make a new problem with the given size.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode)
        {
            this._size = size;
            this._mode = mode;
            resetData();
        }

        /// <summary>
        /// return a copy of the cities in this problem. 
        /// </summary>
        /// <returns>array of cities</returns>
        public City[] GetCities()
        {
            City[] retCities = new City[Cities.Length];
            Array.Copy(Cities, retCities, Cities.Length);
            return retCities;
        }

        /// <summary>
        /// draw the cities in the problem.  if the bssf member is defined, then
        /// draw that too. 
        /// </summary>
        /// <param name="g">where to draw the stuff</param>
        public void Draw(Graphics g)
        {
            float width  = g.VisibleClipBounds.Width-45F;
            float height = g.VisibleClipBounds.Height-45F;
            Font labelFont = new Font("Arial", 10);

            // Draw lines
            if (bssf != null)
            {
                // make a list of points. 
                Point[] ps = new Point[bssf.Route.Count];
                int index = 0;
                foreach (City c in bssf.Route)
                {
                    if (index < bssf.Route.Count -1)
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[index+1]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    else 
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[0]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);
                }

                if (ps.Length > 0)
                {
                    g.DrawLines(routePenStyle, ps);
                    g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width - 1, (float)Cities[0].Y * height - 1, CITY_ICON_SIZE + 2, CITY_ICON_SIZE + 2);
                }

                // draw the last line. 
                g.DrawLine(routePenStyle, ps[0], ps[ps.Length - 1]);
            }

            // Draw city dots
            foreach (City c in Cities)
            {
                g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
            }

        }

        /// <summary>
        ///  return the cost of the best solution so far. 
        /// </summary>
        /// <returns></returns>
        public double costOfBssf ()
        {
            if (bssf != null)
                return (bssf.costOfRoute());
            else
                return -1D; 
        }

        /// <summary>
        ///  solve the problem.  This is the entry point for the solver when the run button is clicked
        /// right now it just picks a simple solution. 
        /// </summary>
        public void solveProblem()
        {
            int x;
            Route = new ArrayList(); 
            // this is the trivial solution. 
            for (x = 0; x < Cities.Length; x++)
            {
                Route.Add( Cities[Cities.Length - x -1]);
            }
            // call this the best solution so far.  bssf is the route that will be drawn by the Draw method. 
            bssf = new TSPSolution(Route);
            // update the cost of the tour. 
            Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();
            // do a refresh. 
            Program.MainForm.Invalidate();

        }

        /**
         * Solve the TSP using a simple greedy algorithm, choose the shortest path at each choice.
         * If we get stuck, backtrack and choose the next best one.
         */
        public void solveGreedy()
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();

            getGreedyRoute();

            timer.Stop();

            bssf = new TSPSolution(Route);
            // update the cost of the tour. 
            Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();
            Program.MainForm.tbElapsedTime.Text = timer.Elapsed.ToString();
            // do a refresh. 
            Program.MainForm.Invalidate();
            return;
        }

        /**
         * Finds a greedy solution, storing the path in Route.
         * 
         * The path will be complete.
         */
        private void getGreedyRoute()
        {
            double distance = double.PositiveInfinity;

            while (double.IsPositiveInfinity(distance))
            {
                // Pick a random starting point, add it to the route
                Route.Clear();
                City startCity = Cities[trueRand.Next(Cities.Length)];
                Route.Add(startCity);

                while (Route.Count != Cities.Length)
                {
                    int minCity = -1;

                    for (int i = 0; i < Cities.Length; i++)
                    {
                        if (Route.Contains(Cities[i]))
                            continue;

                        if (minCity == -1)
                            minCity = i;
                        else if (((City)Route[Route.Count - 1]).costToGetTo(Cities[i]) <
                                 ((City)Route[Route.Count - 1]).costToGetTo(Cities[minCity]))
                            minCity = i;
                    }

                    Route.Add(Cities[minCity]);
                }

                TSPSolution sol = new TSPSolution(Route);
                distance = sol.costOfRoute();
            }
        }

        /**
         * Solve the TSP using a random choice stragegy. Pick a random city, and then
         * pick random edges from there. If we get stuck, start over with a new random 
         * city.
         */
        public void solveRandom()
        {
            Route = new ArrayList();

            City startCity = Cities[trueRand.Next(Cities.Length)];
            Route.Add(startCity);

            int failCount = 0;
            Stopwatch timer = new Stopwatch();
            timer.Start();

            while (Route.Count < Cities.Length)
            {
                City nextCity = Cities[trueRand.Next(Cities.Length)];
                if (Route.Contains(nextCity))
                    continue;

                if (((City)Route[Route.Count - 1]).costToGetTo(nextCity) == Double.PositiveInfinity)
                {
                    failCount++;
                    if (failCount > Cities.Length)
                    {
                        // Start over
                        failCount = 0;
                        Route.Clear();
                        startCity = Cities[trueRand.Next(Cities.Length)];
                        Route.Add(startCity);
                    }

                    continue;
                }
                else
                {
                    Route.Add(nextCity);
                    if (Route.Count == Cities.Length)
                    {
                        timer.Stop();

                        bssf = new TSPSolution(Route);
                        // update the cost of the tour. 
                        Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();
                        Program.MainForm.tbElapsedTime.Text = timer.Elapsed.ToString();
                        // do a refresh. 
                        Program.MainForm.Invalidate();
                        return;
                    }
                }
            }
        }

        /**
         * Solve the TSP using an include/exclude B&B strategy
         */
        public void solveBranchAndBound()
        {
            
        }

        /**
         * Custom TSP solver - Uses Simulated Annealing
         * 
         * Ben implemented the base algorithm using this page as a guide:
         * - http://www.codeproject.com/Articles/26758/Simulated-Annealing-Solving-the-Travelling-Salesma
         * 
         * Revisions:
         * 20 Nov 2012 - Ben - Implemented Base SA algorithm
         * 
         * 
         */
        public void solveCustom()
        {
            // Keep track of how many iterations we did
            int iteration = -1;

            // variables for SA
            double temperature = 10000.0;
            
            // Higher temperature takes longer to run, but yields better results on larger city counts.
            //double temperature = 100000.0;
            
            double absoluteTemp = 0.00001;
            double coolingRate = 0.9999;
            double deltaDistance = 0;
            double distance = 0;

            ArrayList alternateRoute = null;
            TSPSolution alternateSolution = null;

            // First, we need to have a solution to start with. Right now 
            // it's just taking a greedy solution. This may be one place that 
            // we could implement an optimization?
            getGreedyRoute();

            // Generate a new solution using the Greedy route
            bssf = new TSPSolution(Route);
            distance = bssf.costOfRoute();

            Stopwatch timer = new Stopwatch();
            timer.Start();

            while (temperature > absoluteTemp)
            {
                // Get an alternate solution and calculate the difference in their costs
                alternateRoute = getAlternateRouteSA(Route);
                alternateSolution = new TSPSolution(alternateRoute);

                deltaDistance = alternateSolution.costOfRoute() - distance;

                // If the cost is less, OR if the distance is larger, BUT satifsfies the 
                // Boltzmann condition, then we accept the new arrangement.
                if ((deltaDistance < 0) ||
                    (distance > 0 && Math.Exp(-deltaDistance / temperature) > trueRand.NextDouble()))
                {
                    // Accept this new arrangement
                    Route = alternateRoute;

                    distance = distance + deltaDistance;
                }

                // Cool down the temperature
                temperature *= coolingRate;

                iteration++;
            }

            timer.Stop();

            bssf = new TSPSolution(Route);

            // update the cost of the tour. 
            Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();

            Program.MainForm.tbElapsedTime.Text = timer.Elapsed.ToString();
            // do a refresh. 
            Program.MainForm.Invalidate();
            return;
        }

        /**
         * Returns a random variation on route that is still correct. (assuming that 
         * route was complete to begin with)
         * 
         * Used in the SA algorithm.
         * 
         * Method: This is a variation of the 2-opt stragegy. The algorithm picks two edges 
         * at random, and then shifts the path so that their endpoints are connected edges. 
         * For example, if you pick two edges, [a,b] and [c,d], this will return the route where
         * rather than [a,b] and [c,d], it has [a,c] and [b,d]. 
         * 
         */  
        private ArrayList getAlternateRouteSA(ArrayList route)
        {
            ArrayList altRoute = new ArrayList(route);

            // We can't do this for less than 4 cities
            if (route.Count < 4)
                return altRoute;

            // Pick two random edges to cut out and reconnect.
            int e1 = 0, e2 = 0; // Starting cities of both edges, (indices in Route)
            while (e2 - e1 < 2)
            {
                e1 = trueRand.Next(route.Count);
                e2 = trueRand.Next(route.Count);
            }

            // Make sure that e1 always refers to the earlier edge.
            if (e1 > e2)
            {
                int temp = e1;
                e1 = e2;
                e2 = temp;
            }

            // We have the two edges now that we're going to mess with, so now we need 
            // to adjust the route to correctly describe the new path.
            // This involves switching the edges so that [a,b] and [c,d] become [a,c] and [b,d], 
            // but also making sure that we adjust the order of the cities between c and b.
            //
            // For example, if we have a route: [1,2,3,4,5,6,7,8], and we choose edges [2,3] and [7,8] 
            // to switch. Well, if we switch them and then stop, we'll have [1,2,7,4,5,6,3,8]. 
            // At first that may seem ok, but if you draw a picture, you'll see we actually want
            // this: [1,2,7,6,5,4,3,8]
            //
            // We'll do them at the same time by just flipping the entire route between b and c.
            // Until e1 and e2 equal eachother, or they cross, we just march towards the middle, 
            // swapping cities in the route as we go.
            City swapMe = null;
            while (e1 != e2 && e1 < e2)
            {
                swapMe = (City)altRoute[e1];
                altRoute[e1] = altRoute[e2];
                altRoute[e2] = swapMe;

                e1++;
                e2--;
            }

            return altRoute;
        }

        #endregion
    }

}

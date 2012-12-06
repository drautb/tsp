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

        /// <summary>
        ///  this keeps a list of all the neighboring cities for a certain city
        /// </summary>

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
            Program.MainForm.tbElapsedTime.Text = timer.Elapsed.TotalSeconds.ToString();
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
            cityToRoute = new int[Cities.Length];
            routeToCity = new int[Cities.Length];
            temp_cityToRoute = new int[Cities.Length];
            temp_routeToCity = new int[Cities.Length];
            double distance = double.PositiveInfinity;
            int seed;

            while (double.IsPositiveInfinity(distance))
            {
                // Pick a random starting point, add it to the route
                Route.Clear();
                seed = trueRand.Next(Cities.Length);
                City startCity = Cities[seed];
                routeToCity[Route.Count] = seed;
                cityToRoute[seed] = Route.Count;
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
                    routeToCity[Route.Count] = minCity;
                    cityToRoute[minCity] = Route.Count;
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
                        Program.MainForm.tbElapsedTime.Text = timer.Elapsed.TotalSeconds.ToString();
                        // do a refresh. 
                        Program.MainForm.Invalidate();
                        return;
                    }
                }
            }
        }

        /**
        * Solve the TSP using an include/exclude B&B strategy
        * 
        * First, get an upper bound, bssf.
        * 
        * Generate the initial Reduced Cost Matrix, (RCM) which
        * will give us the initial lower bound.
        * 
        * 
        */
        public void solveBranchAndBound()
        {
            // Stats
            int totalStatesGenerated = 0, statesPruned = 0, maxStatesStored = 0;
            bool initialBssfIsFinal = true;

            // Get an upper bound
            getGreedyRoute();
            bssf = new TSPSolution(Route);

            // Generate the RCM, and get it's lower bound from the reduction
            double[,] rcm = generateRCM();
            double lowerBound = reduceCM(ref rcm);

            // Now we need to start throwing states on the queue and processing them..
            PriorityQueue<BBState> stateQueue = new PriorityQueue<BBState>();

            // Create initial state
            BBState initialState = new BBState(rcm, lowerBound);
            stateQueue.Enqueue(initialState, initialState.bound);
            totalStatesGenerated = maxStatesStored = 1;

            // Ok, now we kick off the process!
            // Start the timer..
            Stopwatch timer = new Stopwatch();
            timer.Start();

            BBState curState = null;
            while (stateQueue.Count > 0)
            {
                /*if (timer.ElapsedMilliseconds > 30000) // 30 seconds
                    break; */

                curState = stateQueue.Dequeue();

                // If this state's lower bound is greater than BSSF, then we 
                // prune it out
                if (curState.bound > costOfBssf())
                {
                    statesPruned++;
                    continue;
                }

                // If it's not, then see if it's a complete solution. If it is,
                // then update BSSF.
                if (curState.isCompleteSolution(ref Cities))
                {
                    bssf = new TSPSolution(curState.getRoute(ref Cities));
                    initialBssfIsFinal = false;
                    continue;
                }

                // If it's not a complete solution, but it's within range, then
                // expand it into two child states, one with an included edge, one 
                // with the same edge excluded.
                Tuple<int, int> edge = curState.getNextEdge();

                if (edge == null)
                    continue;

                // Ok, now we have the next edge to include and exclude in different states to maximize 
                // the difference in bounds. So we need to create states corresponding to each and put
                // them in the queue.
                BBState incState = new BBState(curState.cm, curState.bound, curState.includedEdges);
                incState.includedEdges.Add(edge);

                incState.cm[edge.Item1, edge.Item2] = double.PositiveInfinity;
                incState.cm[edge.Item2, edge.Item1] = double.PositiveInfinity;

                for (int t = 0; t < incState.cm.GetLength(0); t++)
                    incState.cm[t, edge.Item2] = double.PositiveInfinity;

                for (int t = 0; t < incState.cm.GetLength(1); t++)
                    incState.cm[edge.Item1, t] = double.PositiveInfinity;

                // Need to take out edges in incState that could be used to complete a premature cycle
                if (incState.includedEdges.Count < incState.cm.GetLength(0) - 1)
                {
                    int start = edge.Item1, end = edge.Item2, city;

                    city = getCityExited(start, incState.includedEdges);
                    while (city != -1)
                    {
                        start = city;
                        city = getCityExited(start, incState.includedEdges);
                    }

                    city = getCityEntered(end, incState.includedEdges);
                    while (city != -1)
                    {
                        end = city;
                        city = getCityEntered(end, incState.includedEdges);
                    }

                    while (start != edge.Item2)
                    {
                        incState.cm[end, start] = double.PositiveInfinity;
                        incState.cm[edge.Item2, start] = double.PositiveInfinity;
                        start = getCityEntered(start, incState.includedEdges);
                    }
                }

                //  finish setting up the state and put it in the queue
                incState.bound = curState.bound + reduceCM(ref incState.cm);

                totalStatesGenerated++;
                if (incState.bound > costOfBssf())
                {
                    statesPruned++;
                }
                else
                {
                    stateQueue.Enqueue(incState, incState.bound);
                    if (stateQueue.Count > maxStatesStored)
                        maxStatesStored = stateQueue.Count;
                }

                BBState exState = new BBState(curState.cm, curState.bound, curState.includedEdges);

                exState.cm[edge.Item1, edge.Item2] = double.PositiveInfinity;
                exState.bound = curState.bound + reduceCM(ref exState.cm);

                totalStatesGenerated++;
                if (exState.bound > costOfBssf())
                {
                    statesPruned++;
                }
                else
                {
                    stateQueue.Enqueue(exState, exState.bound);
                    if (stateQueue.Count > maxStatesStored)
                        maxStatesStored = stateQueue.Count;
                }
            }

            timer.Stop();

            Program.MainForm.tbElapsedTime.Text = " " + timer.Elapsed;

            // update the cost of the tour. 
            Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();
            // do a refresh. 
            Program.MainForm.Invalidate();

            String msg = "\tB&B RESULTS\n";
            if (timer.ElapsedMilliseconds > 30000)
                msg += "\nSearch timed out, 30 seconds expired.";
            if (initialBssfIsFinal)
                msg += "\nInitial BSSF (Greedy) is final solution.";
            else
                msg += "\nA better BSSF than the initial was found.";

            msg += "\n\n\tSTATS:";
            msg += "\nTotal States Created:\t" + totalStatesGenerated;
            msg += "\nTotal States Pruned:\t" + statesPruned;
            msg += "\nMax States Stored: \t" + maxStatesStored;

            MessageBox.Show(msg);

            return;
        }

        private int getCityEntered(int cityExited, List<Tuple<int, int>> edges)
        {
            foreach (Tuple<int, int> t in edges)
            {
                if (t.Item1 == cityExited)
                    return t.Item2;
            }

            return -1;
        }

        private int getCityExited(int cityEntered, List<Tuple<int, int>> edges)
        {
            foreach (Tuple<int, int> t in edges)
            {
                if (t.Item2 == cityEntered)
                    return t.Item1;
            }

            return -1;
        }

        /**
         * Method to generate a reduced cost matrix for the B&B
         */
        private double[,] generateRCM()
        {
            double[,] rcm = new double[Cities.Length, Cities.Length];
            for (int i = 0; i < Cities.Length; i++)
            {
                for (int j = 0; j < Cities.Length; j++)
                {
                    // Cities can't go to themselves
                    if (i == j)
                    {
                        rcm[i, j] = double.PositiveInfinity;
                        continue;
                    }

                    rcm[i, j] = Cities[i].costToGetTo(Cities[j]);
                }
            }

            return rcm;
        }

        /**
         * Method that will reduce a given cost matrix
         */
        public static double reduceCM(ref double[,] cm)
        {
            double minPath = double.PositiveInfinity;
            double lowerBound = 0;

            // Reduce the rows
            for (int i = 0; i < cm.GetLength(0); i++)
            {
                minPath = double.PositiveInfinity;
                for (int j = 0; j < cm.GetLength(1); j++)
                {
                    if (cm[i, j] < minPath)
                        minPath = cm[i, j];
                }

                // if the minimum path found was infinite or 0, then we skip the row. (it's 
                // already reduced or not reducible.
                if (minPath == double.PositiveInfinity || minPath == 0)
                    continue;

                // Otherwise, we need to subtract minPath from each entry in the row
                lowerBound += minPath;
                for (int j = 0; j < cm.GetLength(1); j++)
                {
                    cm[i, j] -= minPath;
                }
            }

            // Reduce the columns
            for (int j = 0; j < cm.GetLength(1); j++)
            {
                minPath = double.PositiveInfinity;

                for (int i = 0; i < cm.GetLength(0); i++)
                {
                    if (cm[i, j] < minPath)
                        minPath = cm[i, j];
                }

                if (minPath == double.PositiveInfinity || minPath == 0)
                    continue;

                lowerBound += minPath;
                for (int i = 0; i < cm.GetLength(0); i++)
                {
                    cm[i, j] -= minPath;
                }
            }

            return lowerBound;
        }

        /**
         * Debug method to print the RCM to the output window
         */
        public static void outputRCM(double[,] rcm)
        {
            for (int i = 0; i < rcm.GetLength(0); i++)
            {
                string line = "";
                for (int j = 0; j < rcm.GetLength(1); j++)
                {
                    line += rcm[i, j].ToString().PadLeft(10, ' ') + "\t";
                }
                Debug.WriteLine(line);
            }
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
            // test the findNeighbors method
            findNeighbors();

            // Keep track of how many iterations we did
            int iteration = -1;

            // variables for SA
            double temperature = 10.0;
            
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

            solveRandomNearNeighbor();

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
                    //Debug.WriteLine("found better solution!");
                    // Accept this new arrangement
                    Route = alternateRoute;
                    Array.Copy(temp_cityToRoute, cityToRoute, cityToRoute.Length);
                    Array.Copy(temp_routeToCity, routeToCity, routeToCity.Length);

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

            Program.MainForm.tbElapsedTime.Text = timer.Elapsed.TotalSeconds.ToString();
            // do a refresh. 
            Program.MainForm.Invalidate();
            return;
        }

        public void solveFurthest()
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();

            ArrayList cur_cities = new ArrayList(Cities);
            Route = new ArrayList();
            City centerPoint, furthestPoint;

            constructInitial(cur_cities);

            while (cur_cities.Count > 0)
            {
                //furthestPoint = findTotalFarthest(cur_cities);
                
                centerPoint = center();
                furthestPoint = findFarthest(centerPoint, cur_cities);
                insertEdge(furthestPoint, cur_cities);
                cur_cities.Remove(furthestPoint);
            }

            bssf = new TSPSolution(Route);
            timer.Stop();
            Program.MainForm.tbCostOfTour.Text = " " + bssf.costOfRoute();
            Program.MainForm.tbElapsedTime.Text = timer.Elapsed.TotalSeconds.ToString();
            Program.MainForm.Invalidate();
        }

        private void insertEdge(City furthestPoint, ArrayList in_cities)
        {
            int prevCity = -1, postCity = -1;
            double min_dist = double.PositiveInfinity;
            double distToEdge;

            for (int j = 0; j < Route.Count; j++)
            {
                if (j < Route.Count - 1)
                {
                    distToEdge = FindDistanceToSegment(furthestPoint, (City)Route[j], (City)Route[j + 1]);
                    //Debug.WriteLine(j + "\tto\t" + (j + 1) + "\t:" + distToEdge);

                    if (distToEdge < min_dist && furthestPoint.costToGetTo((City)Route[j + 1]) != double.PositiveInfinity
                        && ((City)Route[j]).costToGetTo(furthestPoint) != double.PositiveInfinity)
                    {
                        min_dist = distToEdge;
                        prevCity = j;
                        postCity = j + 1;
                    }
                }
                else
                {
                    distToEdge = FindDistanceToSegment(furthestPoint, (City)Route[j], (City)Route[0]);
                    //Debug.WriteLine(j + "\tto\t0\t:" + distToEdge);

                    if (distToEdge < min_dist && furthestPoint.costToGetTo((City)Route[0]) != double.PositiveInfinity
                        && ((City)Route[j]).costToGetTo(furthestPoint) != double.PositiveInfinity)
                    {
                        min_dist = distToEdge;
                        prevCity = j;
                        postCity = 0;
                    }
                }   
            }

            if (prevCity == -1)
            {
                Debug.WriteLine("UH OH! no valid edge has been found, no entry point for new city");
            }
            else
            {
                Route.Insert(prevCity + 1, furthestPoint);
            }
            //in_cities.Remove(furthestPoint);
        }

        private double FindDistanceToSegment(City pt, City p1, City p2)
        {
            City closest;
            double dx = p2.X - p1.X;
            double dy = p2.Y - p1.Y;
            double delev = p2.elevation - p1.elevation;

            if ((dx == 0) && (dy == 0) && (delev == 0))
            {
                // It's a point not a line segment.
                closest = p1;
                dx = pt.X - p1.X;
                dy = pt.Y - p1.Y;
                delev = pt.elevation - p1.elevation;
                return Math.Sqrt(dx * dx + dy * dy + delev * delev);
            }

            // Calculate the t that minimizes the distance.
            double t = ((pt.X - p1.X) * dx + (pt.Y - p1.Y) * dy + (pt.elevation - p1.elevation) * delev) / (dx * dx + dy * dy + delev * delev);

            // See if this represents one of the segment's
            // end points or a point in the middle.
            if (t < 0)
            {
                closest = new City(p1.X, p1.Y, delev);
                dx = pt.X - p1.X;
                dy = pt.Y - p1.Y;
                delev = pt.elevation - p1.elevation;
            }
            else if (t > 1)
            {
                closest = new City(p2.X, p2.Y, delev);
                dx = pt.X - p2.X;
                dy = pt.Y - p2.Y;
                delev = pt.elevation - p2.elevation;
            }
            else
            {
                closest = new City(p1.X + t * dx, p1.Y + t * dy, p1.elevation + t * delev);
                dx = pt.X - closest.X;
                dy = pt.Y - closest.Y;
                delev = pt.elevation - closest.elevation;
            }

            return Math.Sqrt(dx * dx + dy * dy + delev * delev);
        }

        private void constructInitial(ArrayList in_cities)
        {
            findFarthest(in_cities);
        }

        private City center()
        {
            double x = 0, y = 0, elev = 0;
            for (int i = 0; i < Route.Count; i++)
            {
                x += ((City)Route[i]).X;
                y += ((City)Route[i]).Y;
                elev += ((City)Route[i]).elevation;
            }

            elev = elev / Route.Count;
            x = x / Route.Count;
            y = y / Route.Count;

            City temp_city = new City(x, y, elev);
            return temp_city;
        }

        private City findTotalFarthest(ArrayList in_cities)
        {
            int i, j, farthestPoint = -1;
            double temp_dist = 0, dist_agg = 0, max_dist = double.NegativeInfinity;

            for (i = 0; i < in_cities.Count; i++)
            {
                for (j = 0; j < Route.Count; j++)
                {
                    temp_dist = ((City)Route[j]).costToGetTo((City)in_cities[i]);
                    if (temp_dist != double.PositiveInfinity)
                    {
                        dist_agg += temp_dist;
                    }
                }

                if (temp_dist > max_dist)
                {
                    max_dist = temp_dist;
                    farthestPoint = i;
                }

                temp_dist = 0;
                dist_agg = 0;
            }

            //Debug.WriteLine(farthestPoint + "\t" + max_dist);
            return (City)in_cities[farthestPoint];
        }

        private City findFarthest(City in_city, ArrayList in_cities)
        {
            double temp_dist, max_dist = double.NegativeInfinity;
            int max_to = -1;

            for (int i = 0; i < in_cities.Count; i++)
            {
                temp_dist = in_city.costToGetTo((City)in_cities[i]);
                if (temp_dist > max_dist)
                {
                    max_dist = temp_dist;
                    max_to = i;
                }
            }

            //Debug.WriteLine(max_to + "\t" + max_dist);
            return (City)in_cities[max_to];
        }

        private void findFarthest(ArrayList in_cities)
        {
            int i, j, max_from = -1, max_to = -1;
            double max_dist = double.NegativeInfinity;
            double temp_dist;

            for (i = 0; i < in_cities.Count; i++)
            {
                for (j = 0; j < in_cities.Count; j++)
                {
                    if (i != j)
                    {
                        temp_dist = ((City)in_cities[i]).costToGetTo((City)in_cities[j]);
                        if (temp_dist > max_dist && temp_dist != double.PositiveInfinity)
                        {
                            max_dist = temp_dist;
                            max_from = i;
                            max_to = j;
                        }
                    }
                }
            }

            if (max_from != -1 && max_to != -1)
            {
                Route.Add((City)in_cities[max_from]);
                Route.Add((City)in_cities[max_to]);

                in_cities.Remove(Cities[max_from]);
                in_cities.Remove(Cities[max_to]);
            }
            else
            {
                Debug.WriteLine("could not find the maximum length?");
            }
        }

        double max_x, min_x, max_y, min_y;
        int dim_x, dim_y;
        double SCALE_FACTOR = 1000;
        double x_inc, y_inc;
        private ArrayList[,] neighbors;
        private int[] neighbor_x, neighbor_y;
        int[] cityToRoute, routeToCity;
        int[] temp_cityToRoute, temp_routeToCity;
        ArrayList[,] city_neighbors;

        /**
         * Solve the TSP using a random choice stragegy. Pick a random city, and then
         * pick random edges from there. If we get stuck, start over with a new random 
         * city.
         */
        public void solveRandomNearNeighbor()
        {
            findNeighbors();
            Route = new ArrayList();
            ArrayList choices;

            int seed = trueRand.Next(Cities.Length);
            City startCity = Cities[seed];
            Route.Add(startCity);

            City nextCity = startCity;
            int addNext = seed;

            int failCount = 0;
            Stopwatch timer = new Stopwatch();
            timer.Start();

            while (Route.Count < Cities.Length)
            {
                choices = getNeighbors(nextCity);
                nextCity = (City)choices[trueRand.Next(choices.Count)];
                if (Route.Contains(nextCity))
                {
                    //Debug.WriteLine("already in the route!");
                    continue;
                }

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
                        Program.MainForm.tbElapsedTime.Text = timer.Elapsed.TotalSeconds.ToString();
                        // do a refresh. 
                        Program.MainForm.Invalidate();
                        return;
                    }
                }
            }
        }

        /// <summary>
        /// find neighboring cities
        /// </summary>
        /// 
        private void findNeighbors()
        {
            int i, j;
            //double avg_distance = 0, distance, distance_limit;

            //Random rnd = new Random();
            //seed = rnd.Next(Cities.Length);
            //Debug.WriteLine("seed:\t" + seed);
            max_x = double.NegativeInfinity;
            min_x = double.PositiveInfinity;
            max_y = double.NegativeInfinity;
            min_y = double.PositiveInfinity;

            for (i = 0; i < Cities.Length; i++)
            {
                if (Math.Ceiling(Cities[i].X * SCALE_FACTOR) > max_x)
                    max_x = Math.Ceiling(Cities[i].X * SCALE_FACTOR);
                if (Math.Floor(Cities[i].X * SCALE_FACTOR) < min_x)
                    min_x = Math.Floor(Cities[i].X * SCALE_FACTOR);
                if (Math.Ceiling(Cities[i].Y * SCALE_FACTOR) > max_y)
                    max_y = Math.Ceiling(Cities[i].Y * SCALE_FACTOR);
                if (Math.Floor(Cities[i].Y * SCALE_FACTOR) < min_y)
                    min_y = Math.Floor(Cities[i].Y * SCALE_FACTOR);

                //Debug.WriteLine("distance:\t" + distance);
            }

            //Debug.WriteLine("top:\t" + max_x + "\tbottom:\t" + min_x + "\tleft:\t" + min_y + "\tright:\t" + max_y);
            double city_sqrt = Math.Ceiling(Math.Sqrt(Cities.Length)/ 2);
            x_inc = (max_x - min_y) / city_sqrt; //(max_x - min_x) / Cities.Length;
            y_inc = (max_y - min_y) / city_sqrt; //(max_y - min_y) / Cities.Length;
            
            dim_x = dim_y = (int)Math.Ceiling(city_sqrt) + 1;
            int[,] counter = new int[dim_x, dim_y];
            city_neighbors = new ArrayList[dim_x, dim_y];
            neighbors = new ArrayList[dim_x, dim_y];
            neighbor_x = new int[Cities.Length];
            neighbor_y = new int[Cities.Length];

            //ArrayList my_neighbors;
            for (i = 0; i < dim_x; i++)
            {
                for (j = 0; j < dim_y; j++)
                {
                    //Debug.Write(counter[i, j] + "\t");
                    counter[i, j] = 0;
                    neighbors[i, j] = new ArrayList();
                    city_neighbors[i, j] = new ArrayList();
                }
                //Debug.Write("\n");
            }

            //neighbors = new ArrayList[Cities.Length];

            coordinate cur_city;
            //int city_x, city_y;
            for (i = 0; i < Cities.Length; i++)
            {
                cur_city = getCoor(Cities[i]);
                //Debug.WriteLine("city:\t" + i + "\tx:\t" + cur_city.getX() + "\ty:\t" + cur_city.getY());

                city_neighbors[cur_city.getX(), cur_city.getY()].Add(Cities[i]);
                neighbors[cur_city.getX(), cur_city.getY()].Add(i);
                counter[cur_city.getX(), cur_city.getY()] = counter[cur_city.getX(), cur_city.getY()] + 1;
                neighbor_x[i] = cur_city.getX();
                neighbor_y[i] = cur_city.getY();
            }

            //for (i = 0; i < dim_x; i++)
            {
                //for (j = 0; j < dim_y; j++)
                {
                    //if (neighbors[i, j].Count > 0)
                    {
                        //ArrayList choices = getNeighbors((City)city_neighbors[i, j][0]);
                        //Debug.WriteLine(i + "\t" + j + "\t");
                        //for (int l = 0; l < neighbors[i, j].Count; l++)
                        {
                            //Debug.Write(neighbors[i,j].Count + "\t");
                        }
                    }
                }
                //Debug.Write("\n");
            }

            //for (i = 0; i < dim_x; i++)
            {
                //for (j = 0; j < dim_y; j++)
                {
                    //if (neighbors[i, j].Count > 0)
                    {
                        //ArrayList choices = getNeighbors((City)city_neighbors[i, j][0]);
                        //Debug.WriteLine(i + "\t" + j + "\t");
                        //for (int l = 0; l < choices.Count; l++)
                        {
                            //Debug.WriteLine("\t" + ((City)city_neighbors[i, j][0]).costToGetTo((City)choices[l]) + "\t");
                        }
                    }
                }
                //Debug.Write("\n");
            }

            //Debug.WriteLine("x inc:\t" + dim_x + "\ty inc:\t" + dim_y);
            //Debug.WriteLine("Distance Limit:\t" + distance_limit);

        }

        private void printNeighbor(City in_city)
        {
            coordinate coor = getCoor(in_city);
            int x, y;
            x = coor.getX();
            y = coor.getY();

            Debug.WriteLine("neighbors for:\t" + x + "\t" + y);
            for(int i = 0; i < neighbors[x, y].Count; i++)
            {
                Debug.WriteLine(neighbors[x, y][i]);
            }
            Debug.WriteLine("******");
        }

        private class coordinate
        {
            public coordinate(int in_x, int in_y)
            {
                x = in_x;
                y = in_y;
            }

            public int getX()
            {
                return x;
            }

            public int getY()
            {
                return y;
            }

            private int x, y;
        }

        private coordinate getCoor(City in_city)
        {
            int x = (int)(Math.Floor((in_city.X * SCALE_FACTOR - min_x) / x_inc));
            int y = (int)(Math.Floor((in_city.Y * SCALE_FACTOR - min_y) / y_inc));
            coordinate coor = new coordinate(x, y);

            return coor;
        }

        private ArrayList getNeighbors(int city_index)
        {
            ArrayList cur_neighbors = new ArrayList();
            int x = neighbor_x[city_index];
            int y = neighbor_y[city_index];

            // add yourself
            cur_neighbors.AddRange(neighbors[x, y]);
            cur_neighbors.Remove(city_index);

            // add the neighbors
            if (x + 1 < dim_x)
            {
                cur_neighbors.AddRange(neighbors[x + 1, y]);
                if (y + 1 < dim_y)
                    cur_neighbors.AddRange(neighbors[x + 1, y + 1]);
                if (y - 1 >= 0)
                    cur_neighbors.AddRange(neighbors[x + 1, y - 1]);
            }

            if (x - 1 >= 0)
            {
                cur_neighbors.AddRange(neighbors[x - 1, y]);
                if (y + 1 < dim_y)
                    cur_neighbors.AddRange(neighbors[x - 1, y + 1]);
                if (y - 1 >= 0)
                    cur_neighbors.AddRange(neighbors[x - 1, y - 1]);
            }

            if (y - 1 >= 0)
                cur_neighbors.AddRange(neighbors[x, y - 1]);
            if (y + 1 < dim_y)
                cur_neighbors.AddRange(neighbors[x, y + 1]);

            //Debug.WriteLine("x:\t" + x + "\ty:\t" + y + "\tcount:\t" + cur_neighbors.Count);

            return cur_neighbors;
        }

        private ArrayList getNeighbors(City in_city)
        {
            ArrayList cur_neighbors = new ArrayList();
            coordinate coor = getCoor(in_city);
            int x = coor.getX();
            int y = coor.getY();

            // add yourself
            cur_neighbors.AddRange(city_neighbors[x, y]);
            cur_neighbors.Remove(in_city);

            // add the neighbors
            if (x + 1 < dim_x)
            {
                cur_neighbors.AddRange(city_neighbors[x + 1, y]);
                if (y + 1 < dim_y)
                    cur_neighbors.AddRange(city_neighbors[x + 1, y + 1]);
                if (y - 1 >= 0)
                    cur_neighbors.AddRange(city_neighbors[x + 1, y - 1]);
            }

            if (x - 1 >= 0)
            {
                cur_neighbors.AddRange(city_neighbors[x - 1, y]);
                if (y + 1 < dim_y)
                    cur_neighbors.AddRange(city_neighbors[x - 1, y + 1]);
                if (y - 1 >= 0)
                    cur_neighbors.AddRange(city_neighbors[x - 1, y - 1]);
            }

            if (y - 1 >= 0)
                cur_neighbors.AddRange(city_neighbors[x, y - 1]);
            if (y + 1 < dim_y)
                cur_neighbors.AddRange(city_neighbors[x, y + 1]);

            //Debug.WriteLine("x:\t" + x + "\ty:\t" + y + "\tcount:\t" + cur_neighbors.Count);

            return cur_neighbors;
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
            ArrayList choices;
            // We can't do this for less than 4 cities
            if (route.Count < 4)
                return altRoute;

            // Pick two random edges to cut out and reconnect.
            int e1 = 0, e2 = 0; // Starting cities of both edges, (indices in Route)
            //int neighbor_search;   // this decides if a near neighbor is chosen or not
            while (e2 - e1 < 2)
            {
                //neighbor_search = trueRand.Next(2);
                e1 = trueRand.Next(route.Count);
                //printNeighbor((City)altRoute[e1]);
                
                choices = getNeighbors(routeToCity[e1]);
                //for (int i = 0; i < choices.Count; i++)
                {
                    //Debug.WriteLine(choices[i]);
                }
                
                //if (neighbor_search == 1)
                {
                    e2 = (int) choices[trueRand.Next(choices.Count)];
                    //Debug.WriteLine("e1:\t" + e1 + "\tactual city e1:\t" + routeToCity[e1] + "\te2:\t" + e2 + "\tactual e2:\t" + routeToCity[e2]);
                }

                //for (int i = 0; i < cityToRoute.Length; i++)
                {
                    //Debug.WriteLine(i + "\t" + cityToRoute[i] + "\t" + routeToCity[i]);
                }

                //else
                {
                    //e2 = trueRand.Next(route.Count);
                }
                //Debug.WriteLine(e1 + "\t" + neighbors[e1][trueRand.Next(neighbors[e1].Count)]);
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
            e2 = e2 - 1;
            City swapMe = null;
            int swapPosRoute, swapPosCity;
            Array.Copy(routeToCity, temp_routeToCity, routeToCity.Length);
            Array.Copy(cityToRoute, temp_cityToRoute, cityToRoute.Length);
            while (e1 != e2 && e1 < e2)
            {
                swapMe = (City)altRoute[e1];
                swapPosRoute = routeToCity[e1];
                swapPosCity = cityToRoute[routeToCity[e1]];

                altRoute[e1] = altRoute[e2];
                temp_routeToCity[e1] = routeToCity[e2];
                temp_cityToRoute[routeToCity[e1]] = e2;

                altRoute[e2] = swapMe;
                temp_routeToCity[e2] = swapPosRoute;
                temp_cityToRoute[routeToCity[e2]] = swapPosCity;

                e1++;
                e2--;
            }

            //Debug.WriteLine("*****");
            //for (int i = 0; i < cityToRoute.Length; i++)
            {
                //Debug.WriteLine(i + "\t" + temp_cityToRoute[i] + "\t" + temp_routeToCity[i]);
            }

            return altRoute;
        }

        #endregion
    }

}

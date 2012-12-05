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

            Program.MainForm.tbElapsedTime.Text = timer.Elapsed.ToString();
            // do a refresh. 
            Program.MainForm.Invalidate();
            return;
        }

        public void solveFurthest()
        {
            //ArrayList cur_cities = new ArrayList(Cities);
            Route = new ArrayList();

            bssf = new TSPSolution(Route);
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
                        Program.MainForm.tbElapsedTime.Text = timer.Elapsed.ToString();
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

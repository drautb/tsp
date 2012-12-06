using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace TSP
{
    /**
     * BBState represents a state of the TSP Branch and Bound
     * search algorithm.
     */
    class BBState
    {
        // The cost matrix for this state
        public double[,] cm;

        // The lowerBound of this state
        public double bound;

        // List of included edges in this state
        public List<Tuple<int, int>> includedEdges = new List<Tuple<int,int>>();

        public BBState(double[,] cm, double lowerBound)
        {
            this.cm = duplicateCM(cm);
            this.bound = lowerBound;
            this.includedEdges.Clear();
        }

        public BBState(double[,] cm, double lowerBound, List<Tuple<int, int>> includedEdges)
        {
            this.cm = duplicateCM(cm);
            this.bound = lowerBound;
            this.includedEdges = new List<Tuple<int, int>>(includedEdges);
        }

        /**
            * Check to see if this solution is complete and valid
            */
        public bool isCompleteSolution(ref City[] cities)
        {
            //if (includedEdges.Count < cities.Length || includedEdges.Count > cities.Length)
            if (includedEdges.Count >= cities.Length - 1)
                return true;

            return false;
        }

        /**
            * Return the route that this state represents
            */
        public ArrayList getRoute(ref City[] cities)
        {
            if (includedEdges.Count <= 0)
                return null;

            ArrayList route = new ArrayList();
            // Find the right first edge
            Tuple<int, int> startEdge = null;
            foreach (Tuple<int, int> edge in includedEdges)
            {
                startEdge = edge;
                bool isStart = true;
                foreach (Tuple<int, int> checkEdge in includedEdges)
                {
                    if (startEdge.Item1 == checkEdge.Item2)
                    {
                        isStart = false;
                        break;
                    }
                }

                if (isStart)
                    break;
            }

            // start edge is the right start edge here
            // Add the first city
            route.Add(cities[startEdge.Item1]);

            // And the second city
            route.Add(cities[startEdge.Item2]);
            
            int currentCityIdx = startEdge.Item2;

            while (route.Count != cities.Length)
            {
                foreach (Tuple<int, int> edge in includedEdges)
                {
                    if (edge.Item1 == currentCityIdx)
                    {
                        route.Add(cities[edge.Item2]);
                        currentCityIdx = edge.Item2;
                    }

                    if (route.Count >= cities.Length)
                        break;
                }
            }

            return route;
        }

        /**
            * Return the next edge that will maximize the difference between including it and excluding it.
            */
        public Tuple<int, int> getNextEdge()
        {
            Tuple<int, int> bestEdgeSoFar = null;
            double includeBound, excludeBound;
            double difference, maxDifference = -1D;

            // Loop through all edges
            for (int i = 0; i < cm.GetLength(0); i++)
            {
                for (int j = 0; j < cm.GetLength(1); j++)
                {
                    // If this edge isn't eligible (zero cost), skip it.
                    if (cm[i, j] != 0)
                        continue;

                    // Ok, this edge is eligible, calculate the inc/exc difference
                    
                    // For inclusion, make all entries in column i and row j infinite cost.
                    //  then reduce the matrix, adding to the bound as necessary.
                    // -- If this edge is used, later on in the core algorith we need toremember 
                    //      to mark edge [j, i] as infinite as well.
                    double[,] tempCm = duplicateCM(cm);

                    tempCm[i, j] = double.PositiveInfinity;
                    tempCm[j, i] = double.PositiveInfinity;

                    for (int t = 0; t < tempCm.GetLength(0); t++)
                        tempCm[t, j] = double.PositiveInfinity;

                    for (int t = 0; t < tempCm.GetLength(1); t++)
                        tempCm[i, t] = double.PositiveInfinity;

                    includeBound = bound + ProblemAndSolver.reduceCM(ref tempCm);

                    // For exclusion, make the cost of [i, j] infinite, then
                    // b(Se) = b(Sparent) + min(rowi) + min(colj)
                    tempCm = duplicateCM(cm);

                    tempCm[i, j] = double.PositiveInfinity;
                    excludeBound = bound + ProblemAndSolver.reduceCM(ref tempCm);

                    // Calculate the differnce, check to see if this is lower than the lowest so far
                    difference = Math.Abs(excludeBound - includeBound);
                    if (difference > maxDifference)
                    {
                        maxDifference = difference;
                        bestEdgeSoFar = new Tuple<int, int>(i, j);
                    }
                }
            }

            return bestEdgeSoFar;
        }

        /**
            * Helper method to duplicate cost matrices
            */
        public static double[,] duplicateCM(double[,] cm)
        {
            double[,] newCm = new double[cm.GetLength(0), cm.GetLength(1)];

            for (int i = 0; i < cm.GetLength(0); i++)
            {
                for (int j = 0; j < cm.GetLength(1); j++)
                {
                    newCm[i, j] = cm[i, j];
                }
            }

            return newCm;
        }
    }
}

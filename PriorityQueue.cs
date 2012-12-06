//**********************************************************
//* PriorityQueue                                          *
//* Copyright (c) Julian M Bucknall 2004                   *
//* All rights reserved.                                   *
//* This code can be used in your applications, providing  *
//*    that this copyright comment box remains as-is       *
//**********************************************************
//* .NET priority queue class (heap algorithm)             *
//**********************************************************
// Modified for use in CS312 Dec 21, 2010

using System;
using System.Collections;

namespace TSP 
{
    public struct HeapEntry<T>
    {
        private T item;

        public HeapEntry(T item, double priority) : this()
        {
            this.item = item;
            Priority = priority;
        }

        public T Item
        {
            get { return item; }
        }
        public double Priority
        {
            set; get;
        }
    }

    public class PriorityQueue<T>
    {
        private int count;
        private int capacity;
        private HeapEntry<T>[] heap;

        public PriorityQueue()
        {
            capacity = 15; // 15 is equal to 4 complete levels
            heap = new HeapEntry<T>[capacity];
        }

        /// <summary>
        /// This will remove and return the object with the minimum priority (deletemin).
        /// </summary>
        public T Dequeue()
        {
            if (count == 0)
            {
                throw new InvalidOperationException();
            }

            T result = heap[0].Item;
            count--;
            trickleDown(0, heap[count]);
            return result;
        }

        /// <summary>
        /// This will add the object to the queue with the given priority (insert).
        /// </summary>
        public void Enqueue(T item, double priority)
        {
            if (count == capacity)
                growHeap();
            count++;
            bubbleUp(count - 1, new HeapEntry<T>(item, priority));
        }

        #region Private Methods
        private void growHeap()
        {
            capacity = (capacity * 2) + 1;
            HeapEntry<T>[] newHeap = new HeapEntry<T>[capacity];
            System.Array.Copy(heap, 0, newHeap, 0, count);
            heap = newHeap;
        }

        private void bubbleUp(int index, HeapEntry<T> he)
        {
            int parent = (index - 1) / 2;
            // note: (index > 0) means there is a parent
            while ((index > 0) && (heap[parent].Priority > he.Priority))
            {
                heap[index] = heap[parent];
                index = parent;
                parent = (index - 1) / 2;
            }
            heap[index] = he;
        }

        private void trickleDown(int index, HeapEntry<T> he)
        {
            int child = (index * 2) + 1;
            while (child < count)
            {
                if (((child + 1) < count) &&
                    (heap[child].Priority > heap[child + 1].Priority))
                {
                    child++;
                }
                heap[index] = heap[child];
                index = child;
                child = (index * 2) + 1;
            }
            bubbleUp(index, he);
        }

        /// <summary>
        /// Returns the index to the heap of the given item.
        /// </summary>
        private int findItem(T item)
        {
            int retVal = -1;
            for (int i = 0; i < heap.Length; i++)
            {
                if (heap[i].Item.Equals(item)) { retVal = i; break; }
            }
            return retVal;
        }

        #endregion

        /// <summary>
        /// This will change the priority of the given item if the priority given is less
        /// than the current priority.
        /// </summary>
        public void decreaseKey(T item, double priority)
        {
            int index = findItem(item);
            if (index > -1)
            {
                if (priority < heap[index].Priority) { heap[index].Priority = priority; }
                bubbleUp(index, heap[index]);
            }
        }

        /// <summary>
        /// This will return true if the queue contains the given item.
        /// </summary>
        public bool contains(T item)
        {
            return findItem(item) > -1;
        }

        /// <summary>
        /// This will return the priority of the given item.
        /// </summary>
        public double getPriority(T item)
        {
            double retVal = -1;
            int index = findItem(item);
            if (index > -1)
            {
                retVal = heap[index].Priority;
            }
            return retVal;
        }

        public int Count
        {
            get { return count; }
        }

        public T peek()
        {
            if (heap.Length < 1) return default(T);
            return heap[0].Item;
        }
    }
}

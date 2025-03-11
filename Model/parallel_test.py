""" This script tests out parallelization by means of the multiprocessing library. """

import numpy as np
import time
import multiprocessing as mp

def f(x, a, b, c, d):
    print(x,a,b,c,d)
    return 

def f_callback(result):
    return

if __name__ == "__main__":

    iterable = range(8)

    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    t0 = time.time()
    c = 4
    d = "carambar"


    for k in iterable:
        for a in range(8):
            for b in range(8):
                t1 = time.time()
                pool.apply_async(func=f, args=(k,a,b,c,d), callback=f_callback)
                t2 = time.time()

                
        # print("Delta_t for one calculation:", t2-t1)
    t3 = time.time()
    print("Total computation time:", t3-t0)
    pool.close()
    pool.join() # postpones the execution of next line of code until all processes in the queue are done.
    

    # for x in np.arange(3):
    #     print(x)
    #     t1 = time.time()
    #     pool.apply(f, args=(x))
    #     t2 = time.time()
    #     print(t2-t1)

    # pool.close()


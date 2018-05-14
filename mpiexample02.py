#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import numpy as np
from mpi4py import MPI
from math import *
import datetime
from time import time

from parutils import pprint

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

pprint("-"*78)
pprint(" Running on %d cores" % size)
pprint("-"*78)


def calculateY(X):
    value = sin(X[0]) + 7*sin(X[1])**2 + 0.1*(X[2]**4)*sin(X[0])
    return value


N = 1000000
nbr_variable = 3
total_calcul = N * (nbr_variable + 2)
each_calcul = int(ceil(total_calcul / size))

if rank == 0:
    start = time()
    X1_1 = np.random.uniform(low=-pi, high=pi, size=N)
    X2_1 = np.random.uniform(low=-pi, high=pi, size=N)
    X3_1 = np.random.uniform(low=-pi, high=pi, size=N)

    X1_2 = np.random.uniform(low=-pi, high=pi, size=N)
    X2_2 = np.random.uniform(low=-pi, high=pi, size=N)
    X3_2 = np.random.uniform(low=-pi, high=pi, size=N)

    # A has N colones and 3 rows
    A = np.vstack((X1_1, X2_1, X3_1))
    B = np.vstack((X1_2, X2_2, X3_2))
    AB = np.hstack((A, B))

    for j in range(3):
        temp1 = A.copy()
        temp1[j, :] = B[j, :]
        AB = np.hstack((AB, temp1))

    data00 = AB[0, :]
    data01 = AB[1, :]
    data02 = AB[2, :]
else:
    data00 = ''
    data01 = ''
    data02 = ''
data00_short = np.empty(each_calcul, dtype=np.float64)
data01_short = np.empty(each_calcul, dtype=np.float64)
data02_short = np.empty(each_calcul, dtype=np.float64)

comm.Scatter([data00, MPI.DOUBLE], [data00_short, MPI.DOUBLE])
comm.Scatter([data01, MPI.DOUBLE], [data01_short, MPI.DOUBLE])
comm.Scatter([data02, MPI.DOUBLE], [data02_short, MPI.DOUBLE])

AB_short = np.vstack((data00_short, data01_short, data02_short))
#length = AB_short.shape[1]
result_short = np.empty(each_calcul, dtype=np.float64)
for i in range(each_calcul):
    result_short[i] = calculateY(AB_short[:, i])

combine_data = comm.gather(result_short, root=0)

if rank == 0:
    times_actual = datetime.datetime.now()
    stop = time()
    print(str(stop - start) + " seconds")

    temp = np.asarray(combine_data[0])
    for i in range(1, size):
        temp = np.hstack((temp, np.asarray(combine_data[i])))
    #print(temp)
    #np.savetxt(savepath + '/y.csv', results, delimiter=',')
    y = np.reshape(temp, (-1, N))
    #print(y)

    ya = y[0, :]
    #np.savetxt(savepath + '/ya.csv', ya, delimiter=',')
    yb = y[1, :]
    #np.savetxt(savepath + '/yb.csv', yb, delimiter=',')
    ynormal = y[2:, :]
    #np.savetxt(savepath + '/ynormal.csv', ynormal, delimiter=',')

    Vtotal = np.var(combine_data)
    firstorder = []
    Stotal = []

    for i in range(3):
        first = np.mean((ynormal[i, :] - ya) * yb) / Vtotal
        firstorder.append(first)
    print(firstorder)
    #np.savetxt(savepath + '/Sfirstorder.csv', firstorder, delimiter=',')

    for i in range(3):
        total = 0.5 * np.mean((ya - ynormal[i, :]) ** 2) / Vtotal
        Stotal.append(total)
    print(Stotal)
    #np.savetxt(savepath + '/Stotal.csv', Stotal, delimiter=',')




# Allgather data into A again
# this will gather the data and store in all the thread
#comm.Allgather([my_A, MPI.DOUBLE], [A, MPI.DOUBLE])

#pprint("After Allgather:")
#for r in range(comm.size):
#    if comm.rank == r:
#        print("[%d] %s" % (comm.rank, A))
#    comm.Barrier()

#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import numpy as np
from mpi4py import MPI
from math import *

from parutils import pprint

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

pprint("-"*78)
pprint(" Running on %d cores" % size)
pprint("-"*78)

N = 10
nbr_variable = 3
total_calcul = N * (nbr_variable + 2)
each_calcul = int(ceil(total_calcul / size))

if rank == 0:
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
    data02 = AB[0, :]
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

data00_short *= 2
combine_data = comm.gather(data00_short, root=0)

if rank == 0:
    print("collected data in process [%d]" % comm.rank)
    print(combine_data)



# Allgather data into A again
# this will gather the data and store in all the thread
#comm.Allgather([my_A, MPI.DOUBLE], [A, MPI.DOUBLE])

#pprint("After Allgather:")
#for r in range(comm.size):
#    if comm.rank == r:
#        print("[%d] %s" % (comm.rank, A))
#    comm.Barrier()

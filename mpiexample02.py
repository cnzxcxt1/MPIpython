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

begin_index = each_calcul * rank
end_index = each_calcul * (rank + 1)
if rank == (size - 1):
    end_index = N * (nbr_variable + 2)
# tg_0 = 573
# np.random.seed(180428)
if rank == 0:
    X1_1 = [i+1 for i in range(10)]
    X2_1 = [i+2 for i in range(10)]
    X3_1 = [i+3 for i in range(10)]

    X1_2 = [i+4 for i in range(10)]
    X2_2 = [i+5 for i in range(10)]
    X3_2 = [i+6 for i in range(10)]

    # A has N colones and 3 rows
    A = np.vstack((X1_1, X2_1, X3_1))
    B = np.vstack((X1_2, X2_2, X3_2))
    AB = np.hstack((A, B))

    for j in range(3):
        temp1 = A.copy()
        temp1[j, :] = B[j, :]
        AB = np.hstack((AB, temp1))

    data00 = np.empty(total_calcul, dtype=np.float64)
    data00 = AB[0, :]
    data01 = np.empty(total_calcul, dtype=np.float64)
    data01 = AB[1, :]
    data02 = np.empty(total_calcul, dtype=np.float64)
    data02 = AB[2, :]

else:
    data00 = ''
    data01 = ''
    data02 = ''

my_N = end_index - begin_index + 1

my_A = np.empty(my_N, dtype=np.float64)
comm.Scatter([data00[begin_index:(end_index+1)], MPI.DOUBLE], [my_A, MPI.DOUBLE])


# Scatter data into my_A arrays
comm.Scatter([A, MPI.DOUBLE], [my_A, MPI.DOUBLE])

pprint("After Scatter:")
for r in range(comm.size):
    if comm.rank == r:
        print("[%d] %s" % (comm.rank, my_A))
    comm.Barrier()

# Everybody is multiplying by 2
my_A *= 2

# Allgather data into A again
# this will gather the data and store in all the thread
comm.Allgather([my_A, MPI.DOUBLE], [A, MPI.DOUBLE])

pprint("After Allgather:")
for r in range(comm.size):
    if comm.rank == r:
        print("[%d] %s" % (comm.rank, A))
    comm.Barrier()

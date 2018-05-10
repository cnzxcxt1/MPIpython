from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
sendbuf = None
recvbuf = None
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

    data = AB.T
    sendbuf = data
    recvbuf = np.empty([3, 3], dtype='i')
comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    for i in range(size):
        assert np.allclose(recvbuf[i,:], i)
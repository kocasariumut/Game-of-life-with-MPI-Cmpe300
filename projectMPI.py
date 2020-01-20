# Umut Kocasarı
# 2016400177
# Compiling
# Working

#import necessary libraries
from mpi4py import MPI
import numpy as np
import sys
import math
import time

comm = MPI.COMM_WORLD # holds the value of mpi comm_world
rank = comm.Get_rank() # holds which prossess we are
size = comm.Get_size() # holds total number of processes

N = 360 # size of input nxn
T = int(sys.argv[3]) # number of iteration

if rank == 0:

    grid = np.array(open(sys.argv[1]).read().split(), dtype = int).reshape(N, N) # reads the input file, splits it and converts it into a nxn numpy array

    for i in range(size-1): # sends all necessary information
        row = int((i)//math.sqrt(size-1)) # row number of processes which data will be sended
        column = int((i)%math.sqrt(size-1)) # column number of processes which data will be sended
        comm.Send(np.ascontiguousarray(grid[int(row*(N/math.sqrt(size-1))): int(row*(N/math.sqrt(size-1)) + (N/math.sqrt(size-1))), int(column*(N/math.sqrt(size-1))): int(column*(N/math.sqrt(size-1)) + (N/math.sqrt(size-1)))], dtype = int), dest=i+1, tag=10) # sends necessary data to other processes

    for i in range(size-1):
        row = int((i)//math.sqrt(size-1)) # row number of processes which data will be taken
        column = int((i)%math.sqrt(size-1)) # column number of processes which data will be taken
        data = np.empty((int(N/math.sqrt(size-1)), int(N/math.sqrt(size-1))), dtype = int)
        comm.Recv(data, source=i+1, tag=10)
        grid[int(row*(N/math.sqrt(size-1))): int(row*(N/math.sqrt(size-1)) + (N/math.sqrt(size-1))), int(column*(N/math.sqrt(size-1))): int(column*(N/math.sqrt(size-1)) + (N/math.sqrt(size-1)))] = data # takes necessary data from other processes

    np.savetxt(sys.argv[2], grid, fmt = "%d", encoding = "utf-8", newline = " \n") # saves the grid as .txt file given as the argument


else:
    row = int((rank-1)//math.sqrt(size-1)) # row number of process
    column = int((rank-1)%math.sqrt(size-1)) # column number of process
    data = np.empty((int(N/math.sqrt(size-1)), int(N/math.sqrt(size-1))), dtype = int)
    comm.Recv(data, source=0, tag=10) # data taken from process whose rank is 0
    new_grid = np.zeros((int((N/math.sqrt(size-1))) + 2, int((N/math.sqrt(size-1))) + 2), dtype = int) # a grid which is padded by 2 to be able to look all the values around numbers in data even if they are in other processes

    for iteration in range(T): # iterate T times

        new_grid[1:-1, 1:-1] = data # puts the data to the mid of the new_grid

        north = np.empty(data[-1, :].shape, dtype = int)
        south = np.empty(data[0, :].shape, dtype = int)
        west = np.empty(data[:, -1].shape, dtype = int)
        east = np.empty(data[:, 0].shape, dtype = int)
        northeast = np.empty(data[-1, 0].shape, dtype = int)
        southwest = np.empty(data[0, -1].shape, dtype = int)
        northwest = np.empty(data[-1, -1].shape, dtype = int)
        southeast = np.empty(data[0, 0].shape, dtype = int)
        #north part of the processes
        if row%2==0: # first half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[-1, :]), dest= ((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + column + 1, tag=10)
        else:
            comm.Recv(north, source=((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + column + 1, tag=10)
        if row%2==1: # second half of the processes send and others wait
            comm.Send(data[-1, :], dest= ((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + column + 1, tag=10)
        else:
            comm.Recv(north, source=((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + column + 1, tag=10)

        #south part of the processes
        if row%2==0: # first half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[0, :]), dest= ((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + column + 1, tag=10)
        else:
            comm.Recv(south, source=((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + column + 1, tag=10)
        if row%2==1: # second half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[0, :]), dest= ((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + column + 1, tag=10)
        else:
            comm.Recv(south, source=((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + column + 1, tag=10)

        #west part of the processes
        if column%2==0: # first half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[:, -1]), dest= row*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(west, source=row*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        if column%2==1: # second half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[:, -1]), dest= row*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(west, source=row*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)

        #east part of the processes
        if column%2==0: # first half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[:, 0]), dest= row*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(east, source=row*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        if column%2==1: # second half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[:, 0]), dest= row*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(east, source=row*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)

        #northeast part of the processes
        if column%2==0: # first half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[-1, 0]), dest= ((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(northeast, source=((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        if column%2==1: # second half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[-1, 0]), dest= ((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(northeast, source=((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)

        #southwest part of the processes
        if column%2==0: # first half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[0, -1]), dest= ((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(southwest, source=((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        if column%2==1: # second half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[0, -1]), dest= ((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(southwest, source=((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)

        #northwest part of the processes
        if column%2==0: # first half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[-1, -1]), dest= ((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(northwest, source=((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        if column%2==1: # second half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[-1, -1]), dest= ((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(northwest, source=((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)

        #southeast part of the processes
        if column%2==0: # first half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[0, 0]), dest= ((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(southeast, source=((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)
        if column%2==1: # second half of the processes send and others wait
            comm.Send(np.ascontiguousarray(data[0, 0]), dest= ((row+math.sqrt(size-1)-1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+math.sqrt(size-1)-1)%math.sqrt(size-1) + 1, tag=10)
        else:
            comm.Recv(southeast, source=((row+1)%math.sqrt(size-1))*math.sqrt(size-1) + (column+1)%math.sqrt(size-1) + 1, tag=10)


        new_grid[0, 1:-1] = north # puts the north to the correct place in new_grid
        new_grid[-1, 1:-1] = south # puts the south to the correct place in new_grid
        new_grid[1:-1, 0] = west # puts the west to the correct place in new_grid
        new_grid[1:-1, -1] = east # puts the east to the correct place in new_grid
        new_grid[0, -1] = northeast # puts the northeast to the correct place in new_grid
        new_grid[-1, 0] = southwest # puts the southwest to the correct place in new_grid
        new_grid[0, 0] = northwest # puts the northwest to the correct place in new_grid
        new_grid[-1, -1] = southeast # puts the southeast to the correct place in new_grid

        t1 = time.time()
        new_grid_temp = new_grid.copy() # copies it to a new grid because changing the values should not affect others
        for i in range(1, int((N/math.sqrt(size-1)))+1):
            for j in range(1, int((N/math.sqrt(size-1)))+1):
                # game of life part
                counter = new_grid[i+1, j] + new_grid[i-1, j] + new_grid[i, j+1] + new_grid[i, j-1] + new_grid[i+1, j+1] + new_grid[i+1, j-1] + new_grid[i-1, j+1] + new_grid[i-1, j-1]

                if counter < 2:
                    new_grid_temp[i, j] = 0
                elif counter > 3:
                    new_grid_temp[i, j] = 0
                elif counter == 3:
                    new_grid_temp[i, j] = 1
        #print(time.time() - t1)
        data = new_grid_temp[1:-1, 1:-1].copy() # takes the necessary places as the data for next iterations or to send to the manager

    comm.Send(np.ascontiguousarray(data, dtype = int), dest=0, tag=10) # sends data to the manager

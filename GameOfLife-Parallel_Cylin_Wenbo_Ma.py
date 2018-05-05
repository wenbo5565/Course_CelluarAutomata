"""
Parallelized Cylindrical Version for Game oF Life

Wenbo Ma
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import matplotlib.animation as animation
from tqdm import tqdm


# ==========================================
# define functions
# ==========================================
def compute_state(grid):
    """  
        update cell state for a subgrid
        
    parameters:
    -----------
    a subgrid sent a worker. first row and last row are ghost rows

    """
    
    interm_grid = np.copy(grid) # copy input grid
    nrow = grid.shape[0] # number of rows
    ncol = grid.shape[1] # number of coluns
    
    for row_ind in range(1,nrow-1):
        for col_ind in range(1,ncol-1):
            """ summarize neighbor cells state """
            neighbor_state = (grid[row_ind-1,col_ind-1]+grid[row_ind-1,col_ind]+grid[row_ind-1,col_ind+1]
                                +grid[row_ind,col_ind-1]+grid[row_ind,col_ind+1]
                                +grid[row_ind+1,col_ind-1]+grid[row_ind+1,col_ind]+grid[row_ind,col_ind+1])
            """ update state based on Conway's rule """
            if grid[row_ind,col_ind] == 1:
                if neighbor_state < 2 or neighbor_state>3:
                    interm_grid[row_ind,col_ind] = 0
                else:
                    interm_grid[row_ind,col_ind] = 1
            else:
                if neighbor_state == 3:
                    interm_grid[row_ind,col_ind] = 1
                else:
                    interm_grid[row_ind,col_ind] = 0
    return interm_grid
    
def msg_up(grid,dest):
    """
        send up information to previous worker (rank-1)
    """
    comm.send(grid[1,:],dest=dest) # send second row up to rank-1 node
    grid[0,:] = comm.recv(source=dest) # receive info from rank-1 node and put it in first ghost row
    return 0

def msg_down(grid,dest):
    """
        send down information to next worker (rank+1)
    """
    nrow = grid.shape[0] # number of rows
    comm.send(grid[nrow-2,:],dest=dest) # send second row from bottom to next node (rank+1)
    grid[nrow-1,:] = comm.recv(source=dest) # receive info from next node(rank+1) and put it into last row
    return 0
                    
# =========================================
# initilizae the global parameters
# =========================================
nrow = 100 # number of row
ncol = 100 # number of column
prob = 0.5 # theta for Bernoulli distribution
generations = 1000


#===============================================
# initilize subgrid
#===============================================
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
stat = MPI.Status()
subnrow = nrow//size+2 # +2 is ghost row. nrow//size is the number of rows needed to be updated

"""boundary cell is dead"""
subGrid = np.random.binomial(1,prob,size=(subnrow,ncol)) # define subgrid which is held by a single node
subGrid[0,:] = 0 # first row dead
subGrid[subnrow-1,:] =0 # last row dead
subGrid[:,0] = 0 # first column dead
subGrid[:,ncol-1] = 0 # last column dead


ims = [] # initiate a list to store frame

for i in tqdm(range(generations)):
    subGrid=compute_state(subGrid)
    """ message passing section """
    if rank == 0:
        msg_down(subGrid,rank+1) # message passing from/to downward node (rank+1)
        msg_up(subGrid,size-1) # message passing from/to last node (size-1)
    elif rank == size-1:
        msg_up(subGrid,rank-1) # message passing from/to upward node (rank-1)
        msg_down(subGrid,0) # message passing from/to first node (rank 0)
    else:
        msg_up(subGrid,rank-1)
        msg_down(subGrid,rank+1)
    newGrid = comm.gather(subGrid[1:subnrow-1,:],root=0) # gather all updated subgrid to rank 0; only get updated row (no ghost row)    
    if rank == 0:
        result = np.vstack(newGrid) # stack all subgrid
        im = plt.imshow(result,animated=True,interpolation='None')
        ims.append([im])


if rank==0:
    fig = plt.figure()
    print("Present Generation = %d" %(generations))
    ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True,repeat_delay=10)
    plt.show()

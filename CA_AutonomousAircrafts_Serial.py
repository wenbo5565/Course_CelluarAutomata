# This is the file for final project of HPC course
# In this file, I implement serial version of selected paper
# Celluar Automata Modeling of En Route and Arrival Self-Spacing for Autonomous Aircrafts
# written by Charles Kim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import compress

# =======================================================
#                 define class and function
# =======================================================

class airplane():
    """
        an airplane in air transportation system
    
    attributes:
        rank: rank of the airplane in the transportation system. highest rank is 0;
        
        loc: current location of the airplane
        
        depart: departure cell
        
        dest: destinational cell
    
    """
    
    def __init__(self, rank, depart, dest):
        """ return an airplane object """
        
        self.rank = rank
        self.depart = depart
        self.dest = dest
        self.loc = np.copy(depart) # initiate location is departure cell
        self.x = 0 # move in x direction
        self.y = 0 # move in y direction
        
    def plan(self,env):
        """ propose next move 
        
            parameters
                env: the air system grid
        """
        
        """ optimal move """
        self.x = np.sign(self.dest[0] - self.loc[0]) # move direction (1,0,-1) in x-direction
        self.y = np.sign(self.dest[1] - self.loc[1]) # move direction (1,0,-1) in y-direction
        
        
        """ check if optimal move is valid(invalid when it is occupied)
                - if yes, make a move
                - if no, make a move that at least one coordiate is optimal
                - if no, hold at the current location
        """
        if env.grid[self.loc[0]+self.x,self.loc[1]+self.y] == 0:
            return None# optimial move is valid
        print("airplane {0}: optimal invalid, change to suboptimal".format(self.rank))
        if abs(self.x)+abs(self.y) == 1 :
            """ when optimal move is horizontal or vertical """
            if abs(self.x) == 1:
                """ optimal move is [1,0] or [-1,0] """
                if env.grid[self.loc[0]+self.x,self.loc[1]+1] == 0:
                    self.y = 1
                    return None
                elif env.grid[self.loc[0]+self.x,self.loc[1]-1] == 0:
                    self.y = -1
                    return None
            else: 
                """ optimal move is [0,1] or [0,-1] """
                if env.grid[self.loc[0]+1,self.loc[1]+self.y] == 0:
                    self.x = 1
                    return None
                elif env.grid[self.loc[0]-1,self.loc[1]+self.y] == 0:
                    self.x = -1
                    return None
        else:
            """ when optimal move is diagnoal"""
            
            """ check if 2nd order priority is valid """
            if env.grid[self.loc[0]+self.x,self.loc[1]] == 0:
                self.y = 0
                return None
            elif env.grid[self.loc[0],self.loc[1]+self.y] == 0:
                self.x = 0
                return None
            
            elif env.grid[self.loc[0]+self.x,self.loc[1]-self.y] == 0:
                """ check if 3rd order pirority is valid """
                self.y = -1*self.y 
                return None
            elif env.grid[self.loc[0]-self.x,self.loc[1]+self.y] == 0:
                self.x = -1*self.x
                return None
            
        """ no valid cell avaiable. hold at the current cell """
        self.x = 0
        self.y = 0
        print("sub-optimal is invalid, must hold")
        
                
            
        
    def move(self):
        """ move """
        current_loc = np.copy(self.loc)
        self.loc[0] = self.loc[0]+self.x
        self.loc[1] = self.loc[1]+self.y
        return (current_loc,self.loc) # return loc before and after move
    
    def status_check(self):
        """ check if arriving destination """
        if all(self.loc == self.dest):
            return 1
        else:
            return 0
    
# =============================================================================
#     def get_loc(self):
#         """ output current loc and next loc """
#         
# =============================================================================
    
        
class airenv():
    """
        an air-transportation environment (2d grid) comprising of two-dimesional grid
    """
    def __init__(self,nrow,ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.grid = np.zeros((self.nrow,self.ncol))
    
    def update(self,row,col):
        """ update cell in grid """
        self.grid[row,col] = not self.grid[row,col]
    
    def no_fly(self,num):
        """ add no-fly cell in the air system """
        for i in range(num):
            added = False
            while not added:
                x = np.random.choice(self.nrow,1)
                y = np.random.choice(self.ncol,1)
                if self.grid[x,y] == 0:
                    self.grid[x,y] = 2
                    added = True
            
def sys_check(airplanes):
    """
        check if there is any airplane en route
        
    """
    status = []
    for airplane in airplanes:
        status.append(airplane.status_check())
    return np.array(status)
            

        
# =============================================================================
# class airsys():
#     """
#         an air-transportation system comprising of airplane and air-envionment
#     """
#     
#     def __init__(self,nplane,nrow,ncol):
#         """ constructor 
#             
#         parameters
#             nrow: number of rows
#             ncol: number of cols
#             nplane: number of planes
#         """
#         self.nplane = nplane 
#         self.pilots = []
#         self.env = airenv(nrow,ncol)
# =============================================================================
     
    
        
        
# ========================================================
#               implement serial version
# ========================================================
""" set air system parameters """
size = 100 # number of grids for the air transportation system
airsys = airenv(size,size)
nplane = 60 # number of planes
nnofly = 20 # number of no-fly cell
pilots = [] # list of pilots
depart_x = np.random.choice(size,nplane,replace=False) # departure x location
depart_y = np.random.choice(size,nplane,replace=False) # departure y location
dest_x = np.random.choice(size,nplane,replace=False) # dest x location
dest_y = np.random.choice(size,nplane,replace=False) # dest y location

""" create airplanes based on parameters """
for i in range(nplane):
    pilots.append(airplane(i,np.array([depart_x[i],depart_y[i]]),np.array([dest_x[i],dest_y[i]])))
    airsys.update(depart_x[i],depart_y[i]) # departure cell occupied
airsys.no_fly(nnofly) # add 5 no-fly cell
plt.imshow(airsys.grid)
""" update air system """

ims = []
while sys_check(pilots).sum() < nplane:
    """ true if there is at least one plane en route """
    enroute_plane = list(compress(pilots,1-sys_check(pilots))) # get all en route planes
    enroute_plane.sort(key=lambda x: x.rank,reverse=False) # order plane by rank
    for each in enroute_plane:
        """ en_route plane continus to make a move """
        each.plan(airsys)
        loc_info=each.move() # move and get location information
        airsys.update(loc_info[0][0],loc_info[0][1]) # update current cell from occupied to vacant
        airsys.update(loc_info[1][0],loc_info[1][1]) # update next cell from vacant to occupied
    # plt.title("number of en-route flight is {0}".format()))
    im = plt.imshow(airsys.grid,animated=True)
    ims.append([im])
 
len(ims)
fig=plt.figure()
ani = animation.ArtistAnimation(fig,ims, interval=500, blit=True,repeat_delay=1000)
plt.show()
    
    
        
        


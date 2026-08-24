"""Numerical Solver for Single Particle Motion and Particle Collisions"""
import numpy as np
import h5py

#===================Single Particle Motion================================

#====HELPER FUNCTIONS=====
#def to_numpy()  #convert into numpy arrays for computational efficiency
#def integrate() #integrator to find position
#=========================

#Define Initial Conditions

#init_pos = [0,0,0]
#m=1 #electron mass
#q=-1 #charge
#B = [1,0,0] #define magnetic field vector orientation and magntitude

#Solving for single particle, m dv/dt = q (E + v x B)

#v_xi, v_yi, v_zi = ... # current velocity values
#[dv_x, dv_y, dv_z] = q/m * (E+np.cross(v,B))


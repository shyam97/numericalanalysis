# NUMERICAL ANALYSIS OF PDEs (W14014TU)
# -----------------------------------------------------------------------------

# NAME : SHYAM SUNDAR HEMAMALINI
# ROLL : 5071984

# -----------------------------------------------------------------------------
# 2D POISSON EQUATION FOR DOUBLY-UNIFORM GRIDS
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Control the dimensions of the grid and grid resolution here
# THESE CONTROL EVERYTHING BELOW
X = 2
Y = 1
h = 0.2

# Number of grid points along x and y directions
nx = int(X/h)
ny = int(Y/h)

'''Creating Sparse Matrixes for Determination of Lxx________________________'''

maindia_x = (1/h)*np.ones(nx)
belowdia_x = (-1/h) * np.ones(nx)

Dx_diagonals = np.array([maindia_x, belowdia_x]) 

Dx = sp.diags(Dx_diagonals,[0,-1], shape=[nx,nx-1]) #sparse array Dx

Dx_T = Dx.transpose()

#print(Dx.shape)
#print(Dx_T.shape)

Lxx = Dx_T.dot(Dx) #1D Laplacian matrix in x-direction

#print(Lxx.shape)

#plt.figure()
#plt.spy(Lxx)

'''Creating Sparse Matrixes for Determination of Lyy________________________'''

maindia_y = (1/h) * np.ones(ny)
belowdia_y = (-1/h) * np.ones(ny)

Dy_diagonals = np.array([maindia_y, belowdia_y])

Dy = sp.diags(Dy_diagonals,[0,-1], shape=[ny,ny-1]) #sparse array Dy

#print(Dx)

Dy_T = Dy.transpose()

Lyy = Dy_T.dot(Dy) #1D Laplacian matrix in y-direction

#plt.figure()
#plt.spy(Lyy)

'''Creating Identity Matrices_______________________________________________'''

Ix = sp.eye(nx-1)
Iy = sp.eye(ny-1)

'''Kronecker Product to Determine L_________________________________________'''

L = sp.kron(Iy,Lxx) + sp.kron(Lyy,Ix) #from equation (4.12)

print(L.shape)

plt.figure()
plt.spy(L, marker=".", markersize =5, color='blue')

'''Creating grid of x and y coordinates_____________________________________'''

x,y = h*np.mgrid[0:nx+1,0:ny+1]

'''Creating the values of Source Function f_________________________________'''

f = np.zeros((nx-1,ny-1)) #initialising f

for i in range(0,nx-1):
    for j in range(0,ny-1): #generating the values of f
        f[i][j] = 20 * np.sin(np.pi*y[i+1][j+1])* \
                        np.sin(1.5*np.pi*x[i+1][j+1] + np.pi)
    
#print(f)

# Visualising f with imshow()
plt.figure()
ax = plt.gca()
im = ax.imshow(f.T,origin='lower',cmap='jet', extent=[0,2,0,1])
plt.title("Source Function $f$")
plt.ylabel("$y$")
plt.xlabel("$x$")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)

'''Creating Boundary Value Vectors__________________________________________'''
    
boundary_x0 = np.zeros(ny+1) #1D vectors denoting the edges of the grid
boundary_xn = np.zeros(ny+1)
boundary_y0 = np.zeros(nx+1) 
boundary_yn = np.zeros(nx+1)

for i in range(0,nx+1):
    boundary_y0[i] = np.sin(0.5*np.pi*x[i][0]) #creating the values of BCs

for i in range(0,ny+1):
    boundary_x0[i] = np.sin(2*np.pi*y[0][i])
    boundary_xn[i] = np.sin(2*np.pi*y[nx][i])
    
'''Inserting Boundary Values________________________________________________'''

for i in range(0,ny-1):
    f[0][i] = f[0][i] + boundary_x0[i+1] / (h**2)
    f[nx-2][i] = f[nx-2][i] + boundary_xn[i+1] / (h**2)
    
for i in range(0,nx-1):
    f[i][0] = f[i][0] + boundary_y0[i+1] / (h**2)
    
#print(f)
    
'''Reshaping the RHS Matrix_________________________________________________'''
    
F = np.reshape(f,-1,order='F') #reshaping 2D 'f' to 1D 'F'
#print(F.shape)
    
'''Solving the Algebraic Equation Lu=F______________________________________'''

u = la.spsolve(L,F) #sparse solving the matrices L and F
U = np.reshape(u,(nx-1,ny-1),order='F') #reshaping 1D 'u' to 2D 'U'
#print(U.shape)

'''Plotting the solution____________________________________________________'''

plt.figure()
ax = plt.gca()
im = ax.imshow(U.T,origin='lower',cmap='jet',extent=[0,2,0,1])
plt.title("Solution $u$")
plt.ylabel("$y$")
plt.xlabel("$x$")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)

'''========================================================================='''
    
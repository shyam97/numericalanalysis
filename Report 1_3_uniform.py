# NUMERICAL ANALYSIS OF PDEs (W14014TU)
# -----------------------------------------------------------------------------

# NAME : SHYAM SUNDAR HEMAMALINI
# ROLL : 5071984

# -----------------------------------------------------------------------------
# POISSON UNIFORM GRID
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

h = 0.2 #grid refinement based on h
n = int(1/h + 1) #number of nodes including boundaries

# Generating the Laplacian matrix 
L = np.diag(2*np.ones(n-2)) + np.diag(-1*np.ones(n-3),1) + np.diag(-1*\
         np.ones(n-3),-1)
L = L / h**2
print("L: ")

X = np.linspace(0,1,n)  #defining the grid

'''RHS Matrix f1------------------------------------------------------------'''

f1 = np.ones(n-4) #initialising RHS Matrix
f1 = np.insert(f1, 0, 1 + 1/h**2) #inserting boundary values in RHS matrix
f1 = np.insert(f1, n-3, 1 + 2/h**2)

print("f1: ",f1)

u1 = np.linalg.solve(L,f1) #solving L and f1

u1 = np.insert(u1,0,1) #inserting boundary values in solution space
u1 = np.insert(u1,n-1,2)

U1 = -((X**2)/2) + 3*X/2 + 1 #exact solution

print("Numerical Solution u1:", u1) #comparing numerical and exact solutions
print("Exact Solution U1:", U1)

e1 = np.linalg.norm(U1-u1) #global error in U1
print("Norm of u1 = ",e1,'\n')


'''RHS Matrix f2------------------------------------------------------------'''

f2 = np.zeros(n-4) #initialising 

# Assigning values to RHS matrix f2 that are not in the boundary
for i in range(n-4):
    f2[i] = np.exp(h*(i+2)) 

# Inserting boundary values into RHS matrix f2 to the first and last elements
f2 = np.insert(f2, 0, (np.exp(h) + 1/h**2))
f2 = np.insert(f2, n-3, (np.exp(1-h) + 2/h**2))

print("f2: ",f2)

u2 = np.linalg.solve(L,f2) #solving L and f2

u2 = np.insert(u2,0,1) #inserting boundary values in solution space
u2 = np.insert(u2,n-1,2)

U2 = -1*np.exp(X) + np.e*X + 2 #exact solution

print("Numerical Solution u2:", u2) #comparing numerical and exact solutions
print("Exact Solution U2:", U2)

e2 = np.linalg.norm(U2-u2) #global error in U2
print("Norm of u2 = ",e2)

''' Plots of Exact vs Numerical Solution -----------------------------------'''

Y = np.linspace(0,1,100*n) #finer grid for exact solution
U1 = -((Y**2)/2) + 3*Y/2 + 1 #exact solution U1 on finer grid
U2 = -1*np.exp(Y) + np.e*Y + 2 #exact solution U2 on finer grid

# Plot u1 vs U1
plt.figure(1)
plt.subplot(121)
plt.plot(Y,U1,'b-',label='Analytical')
plt.plot(X,u1,'ro',label='Numerical FD')
plt.title("$f_{1}(x)=1$")
plt.ylabel("$u_1{}}(x)$")
plt.xlabel("$x$")
plt.legend()

#Plot u2 vs U2
plt.subplot(122)
plt.plot(Y,U2,'b-',label='Analytical')
plt.plot(X,u2,'ro',label='Numerical FD')
plt.title("$f_{2}(x)=e^x$")
plt.ylabel("$u_2{}}(x)$")
plt.xlabel("$x$")
plt.legend()

''' Grid Refinement --------------------------------------------------------'''

num = 5 #this variable controls the number of refinements

# Initialising error matrices for grid refinement  
E2 = np.zeros(num)

# First grid resolution has already been executed above
E2[0] = e2

# Initialising matrix of grid spacing H
H = np.zeros(num)
H[0] = 0.2

for k in range(1,num):
    
    h = 0.2 / (2**k) #grid refinement based on h
    n = int(1/h + 1) #number of nodes including boundaries
    
    H[k] = h #saving the value of h onto the matrix H
    
    # Generating the Laplacian matrix 
    L = np.diag(2*np.ones(n-2)) + np.diag(-1*np.ones(n-3),1) + np.diag(-1*\
             np.ones(n-3),-1)
    L = L / h**2
    
    X = np.linspace(0,1,n)  #defining the grid

    '''RHS Matrix f2--------------------------------------------------------'''
    
    f2 = np.zeros(n-4) #initialising 
    
    # Assigning values to RHS matrix f2 that are not near the boundary
    for i in range(n-4):
        f2[i] = np.exp(h*(i+2)) 
    
    # Inserting boundary values into RHS matrix f2 to first and last elements
    f2 = np.insert(f2, 0, (np.exp(h) + 1/h**2))
    f2 = np.insert(f2, n-3, (np.exp(1-h) + 2/h**2))
    
    u2 = np.linalg.solve(L,f2) #solving L and f2
    
    u2 = np.insert(u2,0,1) #inserting boundary values in solution space
    u2 = np.insert(u2,n-1,2)
    
    U2 = -1*np.exp(X) + np.e*X + 2 #exact solution
    
    E2[k] = np.linalg.norm(U2-u2) #global error in U2
    

''' Curve fitting for Error vs H -------------------------------------------'''
    
from scipy import optimize

# Plot Error vs H
plt.figure()
plot1 = plt.subplot(111)
plot1.scatter(H, E2, marker = ".", c='r', label = "Error")
plot1.set_ylim([0, 0.0012])
plot1.title.set_text('Error vs h')
plot1.legend(loc='upper left')
plt.xlabel('h')
plt.ylabel('Error')

# Define the power function "error = C.h^Alpha" 
def error(h, c, Alpha):
    return c * (h**Alpha)

# Use optimize.curve_fit to determine the values of C and Alpha
params, params_covariance = optimize.curve_fit(error, H, E2 , p0=[0.01,1.5])
print("C, Alpha: ", params)

K_fine = np.linspace(0,num-1,100*num) #finer grid for plotting fitted curve
H_fine = 0.2 / 2**(K_fine)

# Plot Error vs H with fitted curve
plt.figure()
ax1 = plt.subplot(111)
ax1.plot(H_fine, error(H_fine, params[0], params[1]),
         label='Fitted function')
ax1.scatter(H, E2, c='r', marker='.', label='Error Data')
ax1.set_ylim([0, 0.0012])
ax1.title.set_text('Curve Fitting of Error Data')
ax1.legend(loc='best')
plt.xlabel('h')
plt.ylabel('Error')




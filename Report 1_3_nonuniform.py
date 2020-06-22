# NUMERICAL ANALYSIS OF PDEs (W14014TU)
# -----------------------------------------------------------------------------

# NAME : SHYAM SUNDAR HEMAMALINI
# ROLL : 5071984

# -----------------------------------------------------------------------------
# POISSON NON-UNIFORM GRID
#------------------------------------------------------------------------------
'''
The program has been written in a way that it is flexible. Variables like num &
k can be used to refine the grid as in the uniform grid code, or to control the
skipping positions.

The position where the non-uniformity occurs on the grid can be changed using
the variable "position". It can be set to a fixed value or can be set to change
with the loop (useful for keeping h fixed and varying position alone) or can
also be set to change with the grid (useful for setting the non-uniformity in
the middle of the grid or to a certain x-coordinate).

Comment out the lines of code that are not necessary for that particular
solution.

'''
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

num = 5 #this variable controls the number of loops

K = np.zeros(num) #initialising grid of k values
H = np.zeros(num) #initialising grid of h values

error1 = np.zeros(num) #initialising error matrices
error2 = np.zeros(num)

for k in range(0,num):
    
    K[k] = k + 1 #saving the iteration number to a matrix
    
    ''' Control the value of h here ----------------------------------------'''
    #h = 0.2 #Constant h
    h = 0.2/(2**k) #Variable h with k
    
    N = int(1/h + 1)
    n = int(1/h)
    
    ''' Control the skipping position here ---------------------------------'''
    #position = 1 #Fixed position
    #position = k+1 #Variable position with loop
    position = int(N/2) #Variable position with grid
    
    H[k] = h
    #print("Number of nodes:",n)
    #print("Skipping position:",position)
    #print("h = ",h)
    
    ''' Defining the grid --------------------------------------------------'''    
    X = np.linspace(0,1,N) #grid for numerical solution
    Y = np.linspace(0,1,100*N) #finer grid for exact solution
    
    ''' Defining Laplacian matrix ------------------------------------------'''    
    L=np.diag(2*np.ones(n-2)) + np.diag(-1*np.ones(n-3),1) + np.diag(-1*\
             np.ones(n-3),-1)
    L = L / h**2
    
    ''' Defining the RHS matrices ------------------------------------------'''    
    f1 = np.ones(n-4) #initialising RHS matrices
    f2 = np.ones(n-4)
    
    bc = 1/h**2 #makes it easier to insert in code
    
    ''' If position does not lie on the grid'''
    if(position>n-1 or position<1):
        position = int(input("Enter a value for position in [ 1, n-1]: "))
        
    X = np.delete(X, position) 
     
# If the non-uniformity occurs near the left boundary -------------------------
    
    if(position==1):
        
        L[0][0] = L[0][0] * 0.5 #changing the values in Laplacian matrix
        L[0][1] = L[0][1] * 2 / 3
        
        f1 = np.insert(f1, 0, 1 + bc/3) #inserting the modified boundary values
        f1 = np.insert(f1, n-3, 1 + 2*bc)
        
        for i in range(n-4): #initialising f2
            f2[i] = np.exp(h*(i+3))
        
        f2 = np.insert(f2, 0, (np.exp(2*h) + bc/3)) #inserting the modified BCs
        f2 = np.insert(f2, n-3, (np.exp(1-h) + 2*bc))
        
# If the non-uniformity occurs near the right boundary ------------------------
    
    elif(position==(n-1)):
        
        L[n-3][n-3] = L[n-3][n-3] *0.5 #changing the values in Laplacian matrix
        L[n-3][n-4] = L[n-3][n-4] * 2 / 3
        
        f1 = np.insert(f1, 0, 1 + bc) #inserting the modified boundary values
        f1 = np.insert(f1, n-3, 1 + 2*bc/3)
        
        for i in range(n-4): #initialising f2
            f2[i] = np.exp(h*(i+2))
        
        f2 = np.insert(f2, 0, (np.exp(h) + bc)) #inserting the modified BCs
        f2 = np.insert(f2, n-3, (np.exp(1-2*h) + 2*bc/3))
        
# If the non-uniformity does not occur near the boundaries --------------------
        
    elif(position<(n-1) and position>1):
        
        # Changing the values in Laplacian matrix
        L[position-2][position-2] = L[position-2][position-2] / 2
        L[position-1][position-1] = L[position-1][position-1] / 2
        L[position-2][position-1] = L[position-2][position-1] / 3
        L[position-1][position-2] = L[position-1][position-2] / 3
        
        # Initialising f2 
        for i in range(n-4):
            if(position>i+2):
                f2[i] = np.exp(h*(i+2))
            else:
                f2[i] = np.exp(h*(i+3))
        
        # Inserting the modified boundary values
        
        # Special cases when the non-uniformity occurs at positions 2 or n-2
        
        if(position == 2):
            L[position-1][position] = L[position-1][position] * 2 / 3
            f1 = np.insert(f1, 0, 1 + 2*bc/3)
            f1 = np.insert(f1, n-3, 1 + 2*bc)
            f2 = np.insert(f2, 0, (np.exp(h) + 2*bc/3))
            f2 = np.insert(f2, n-3, (np.exp(1-h) + 2*bc))
            
        elif(position == (n-2)):
            L[position-2][position-3] = L[position-2][position-3] * 2 / 3
            f1 = np.insert(f1, 0, 1 + 1*bc)
            f1 = np.insert(f1, n-3, 1 + 4*bc/3)
            f2 = np.insert(f2, 0, (np.exp(h) + 1*bc))
            f2 = np.insert(f2, n-3, (np.exp(1-h) + 4*bc/3))
            
        else:
            L[position-1][position] = L[position-1][position] * 2 / 3
            L[position-2][position-3] = L[position-2][position-3] * 2 / 3
            f1 = np.insert(f1, 0, 1 + bc)
            f1 = np.insert(f1, n-3, 1 + 2*bc)
            f2 = np.insert(f2, 0, (np.exp(h) + bc))
            f2 = np.insert(f2, n-3, (np.exp(1-h) + 2*bc))
            
    #print("L:\n",L)
    #print("f1: ",f1)
    #print("f2: ",f2,"\n")

# L, f1 and f2 are now properly defined, hence proceeding to solution ---------
        
    u1 = np.linalg.solve(L,f1) #solving for RHS matrix f1
    
    #print("u1:",u1)
    
    u1 = np.insert(u1, 0, 1) #inserting boundary values
    u1 = np.insert(u1, n-1, 2)
    
    #print(u1)
    
    U1 = -((Y**2)/2) + 3*Y/2 + 1 #finer grid
    U1e = -((X**2)/2) + 3*X/2 + 1 #grid with non-uniformity
    
    #print(np.linalg.norm(U1-U1c))
    error1[k] = np.linalg.norm(u1-U1e) #error norm for f1
    #print("Norm of u1, k=",k,":",error1[k])

# -----------------------------------------------------------------------------
    
    u2 = np.linalg.solve(L,f2) #solving for RHS matrix f2
        
    #print("u2: ",u2,"\n")
    
    u2 = np.insert(u2, 0, 1) #inserting boundary values
    u2 = np.insert(u2, n-1, 2)
    
    #print(u2)
    
    U2 = -1*np.exp(Y) + np.e*Y + 2 #finer grid
    U2e = -1*np.exp(X) + np.e*X + 2 #grid with non-uniformity
    
    #print(np.linalg.norm(U2-U2c),"\n")
    error2[k] = np.linalg.norm(u2-U2e) #error norm for f2
    print("Norm (u2), h=",h,", position=",position,":",error2[k])
    
# Plots -----------------------------------------------------------------------
    
    # The solution is plotted only for h = 0.2. Changing the conditions will
    # plot for any required value of h or k.
    
    if (k==0):
        
        # Plot for U1
        plt.figure()
        plt.subplot(121)
        plt.plot(Y,U1,'b-',label='Analytical')
        plt.plot(X,u1,'ro',label='Numerical')
        plt.title("$f_{1}(x)=1$")
        plt.ylabel("$u_1{}}(x)$")
        plt.xlabel("$x$")
        plt.legend()
        
        # Plot for U2
        plt.subplot(122)
        plt.plot(Y,U2,'b-',label='Analytical')
        plt.plot(X,u2,'ro',label='Numerical')
        plt.title("$f_{2}(x)=e^x$")
        plt.ylabel("$u_2{}}(x)$")
        plt.xlabel("$x$")
        plt.legend()

# Analysing the error data ----------------------------------------------------
        
#print(error2)

y_data = error2
x_data = H #use this for comparing errors from grid refinement
#x_data = K #use this for comparing errors from variation in skipping position

# Scatter plot Error vs H or K
plt.figure()
plot1 = plt.subplot(111)
plot1.scatter(x_data, y_data, marker = ".", c='r', label = "Error")
plot1.title.set_text('Error vs h')
plot1.legend(loc='upper left')
plt.xlabel('$h$') # h or k depending on x_data
plt.ylabel('Error')

# Comment out the following section if variation in skipping position is being
# analysed.

# Define the power function "error = C.h^Alpha" 
def error(h, c, Alpha):
    return c * (h**Alpha)

# Use optimize.curve_fit to determine the values of C and Alpha
params, params_covariance = optimize.curve_fit(error, x_data, y_data, p0=[0.01,1.5])
print("C, Alpha: ", params)
#print("x* = L/2, Alpha = ",params[1]) #modify print to print only Alpha

K_fine = np.linspace(0,num-1,100*num) #finer grid to plot fitted curve
H_fine = 0.2/ 2**(K_fine)

# Plot Error vs H with fitted curve
plt.figure()
ax1 = plt.subplot(111)
ax1.plot(H_fine, error(H_fine, params[0], params[1]),
         label='Fitted function')
ax1.scatter(H,error2,c='r',marker='.', label='Error Data')
#ax1.set_ylim([0, 0.0012])
ax1.legend(loc='best')
plt.xlabel('h')
plt.ylabel('Error')

# -----------------------------------------------------------------------------
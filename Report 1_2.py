# NUMERICAL ANALYSIS OF PDEs (W14014TU)
# -----------------------------------------------------------------------------

# NAME : SHYAM SUNDAR HEMAMALINI
# ROLL : 5071984

# -----------------------------------------------------------------------------
# PROPERTIES OF 1D LAPLACIAN OPERATOR
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

n = 11 #number of nodes including the boundary
h = 1/(n-1)

A = 2.0*np.ones(n-2, dtype=int) # n-2 since first and last nodes are boundaries
B = np.diag(A,0)

P = -1.0*np.ones(n-3, dtype=int) # n-3 since the adjacent diagonals have 1 less element
Q = np.diag(P,1)
R = np.diag(P,-1)

L = (1/h**2)*(B + R + Q)

print("First row: ", L[0]) #first row
print("Second row: ", L[1]) #second row
print("Last row: ", L[n-3]) #last row

plt.spy(L, marker=".", color="red") #spy plot

eigval,eigvec = np.linalg.eig(L) #calculating eigenvalues and eigenvectors
eigval.sort() #sorting the eigenvalues from lowest to highest

I = np.linspace(1,n-2,n-2) #generate matrix of numbers from 1 to 9
Eigval = (np.pi*I)**2 #generating the eigenvalues of operator L

print(eigval,Eigval) #comparing eigenvalues

# Plot Analytical vs Numerical Eigenvalues
plt.figure()
plt.plot(I,eigval,'b.-',label='Numerical Values')
plt.plot(I,Eigval,'r.-',label='Analytical Values')
plt.title("Eigenvalues: Numerical vs Analytical")
plt.xlabel("$i$")
plt.ylabel("Eigenvalues")
plt.legend(loc='best')

zeros = np.zeros(n-2) #generate a matrix of zeros
eigvec = np.insert(eigvec,0, zeros,0) #inserting boundaries to eigenvectors
eigvec = np.insert(eigvec,n-1,zeros,0)

X = np.linspace(0,1,n)
Y = np.linspace(0,1,200)

# Plotting Eigenvectors requires sorting --------------------------------------

for i in range(0,n-2):

    for j in range(0,n-2):
            
            # Normalized eigenfunction has a factor of sqrt(2h)
            # Generating (i+1)th eigenfunction below
            Eigvec_c = ((2/(n-1))**0.5)*np.sin(np.pi*X*(i+1))
            
            # Comparing the norm of the eigenfunction with all the eigenvectors
            norm_t = np.linalg.norm(Eigvec_c - eigvec[:,j])
            norm_t2 = np.linalg.norm(Eigvec_c + eigvec[:,j])
            
            # Both - and + are taken because some eigenvectors may be opposite
            # in sign.
            
            if (norm_t<1e-4):
                
                # Generating the eigenfunction for a finer grid
                Eigvec_p = ((2/(n-1))**0.5)*np.sin(np.pi*Y*(i+1)) 
                
                # Plotting the matching eigenfunction and eigenvector
                plt.figure()
                plt.plot(X,eigvec[:,j],'bo',label='Eigenvectors')
                plt.plot(Y,Eigvec_p,'r',label='Eigenfunction')
                plt.title("Eigenvector vs Eigenfunction #%i" %(i+1))
                plt.xlabel("$x$")
                plt.legend(loc='upper right')
                
            if (norm_t2<1e-4):
                
                # Generating the eigenfunction for a finer grid
                Eigvec_p = ((2*h)**0.5)*np.sin(np.pi*Y*(i+1))
                
                # Plotting the matching eigenfunction and eigenvector
                plt.figure()
                plt.plot(X,-1*eigvec[:,j],'bo',label='Eigenvectors')
                plt.plot(Y,Eigvec_p,'r',label='Eigenfunction')
                plt.title("Eigenvector vs Eigenfunction #%i" %(i+1))
                plt.xlabel("$x$")
                plt.legend(loc='upper right')
        
        
        
    

# Reference Scipy Cookbook Solve KdV
# Import libraries
import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
import matplotlib.pyplot as plt
import imvideo as imv
import time

def main():
    '''main numerical KdV solution function'''
    # Set the size of the domain, and create the discretized grid.
    L = 50      #length of periodic boundary
    N = 64      #number of spatial steps
    dx = L / (N - 1.0)      #spatial step size
    x = np.linspace(0, (1-1.0/N)*L, N)      #initialize x spatial axis

    main_start = time.time()    
    # Set the initial conditions.
    # Not exact for two solitons on a periodic domain, but close enough...
    u0 = kdv_soliton_solution(x-0.33*L, 0.75) #+ kdv_soliton_solution(x-0.65*L, 0.4)

    # Set the time sample grid.
    T = 600
    t = np.linspace(0, T, 501)

    sol = solve_kdv(kdv_model, u0, t, L, 5000)
    print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))

    plot(sol, L, T)
    #animate(sol, L, T, 50)
    return

# The KdV model using Fast Fourier Transform
def kdv_model(u, t, L):
    '''The KdV model using FFT
    Input:
            u       (float)     wave amplitude
            t       (float)     simulation duration
            L       (float)     X length for periodic boundaries 
    Output:
            dudt    (float)     left side of the time differential
    '''
    #Compute the space differentials of u
    ux = psdiff(u, period=L)                #1st order differential 
    uxxx = psdiff(u, period=L, order=3)     #3rd order differential 
    
    dudt = -6*u*ux - uxxx #- 0.01*u                  #KdV model; time differential

    return dudt


# The analytical soliton solution to the KdV
def kdv_soliton_solution(x, c):
    '''The exact soliton solution to the KdV
    Input: 
            x   (float)     variable 1
            c   (float)     variable 2
    Output:
            u   (float)     wave amplitude
    '''
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)
    #u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)-0.005*x

    return u

def solve_kdv(model, u0, t, L, steps):
    '''Solve the KdV using Scipy odeint
    Input:
        model               kdv model
        u0      (float/int) initial amplitude
        t       (int)       time range
        L       (int)       periodic range
        steps   (int)       maximum steps allowed
    Output:
        sol     (array)     periodic solutions'''
    sol = odeint(model, u0, t, args=(L,), mxstep=steps)

    return sol

def plot(sol, rangeX, rangeT):
    '''plot the KdV solution
    Input:
        sol
        rangeX      (float/int)
        rangeT      (float/int)
    Output:
        None    
    '''
    plt.figure(figsize=(8,8))
    plt.imshow(sol, extent=[0,rangeX,0,rangeT])     #test sol[::-1, :]
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.axis('auto')
    plt.title('Korteweg-de Vries on a Periodic Domain')
    plt.show()

    return 

def animate(sol, rangeX, rangeT, frames):
    '''imvideo animation test function; 
        Not particularly useful in this application'''
    images = []
    plt.figure(figsize=(8,8))
    fullSolution = sol[::-1, :]
    xSample = fullSolution[0]
    xZeros = xSample
    for i in range(len(xSample)):
        xZeros[i] = 0
    print('total video frames ' + str(len(sol[::-1, :])))

    start_time = time.time()
    for n in range(len(sol[::-1, :])):
        updatedSolution = fullSolution
        updatedSolution[:n] = xZeros
        plt.imshow(updatedSolution, extent=[0,rangeX,0,rangeT])
        #plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.axis('auto')
        plt.title('Korteweg-de Vries on a Periodic Domain')
        #print('frame '+ str(n))
        images = imv.memory.savebuff(plt, images)
        plt.close()

    print('begin video construction')
    revImg = reversed(images)
    imv.memory.construct(revImg, 'matplot_demo.avi', frames)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    return 

if __name__ == "__main__":
    main()

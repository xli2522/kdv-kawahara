# KdV Numerical Stability Analysis
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
    u0 = waveEquation.kdv_soliton_solution(x-0.33*L, 0.75) #+ kdv_soliton_solution(x-0.65*L, 0.4)

    # Set the time sample grid.
    T = 600
    t = np.linspace(0, T, 501)
    dt = len(t)

    sol = waveEquation.solve_kdv(waveEquation.kdv_model, u0, t, L, 5000)
    print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))

    #visual.plot_profile(sol, 250, N)
    visual.plot_video(sol, len(t), N)

    return

class waveEquation:
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
        
        dudt = -6*u*ux - uxxx                   #KdV model; time differential

        return dudt

    # The approximated soliton solution to the KdV
    def kdv_soliton_solution(x, c):
        '''The exact soliton solution to the KdV
        Input: 
                x   (float)     variable 1
                c   (float)     variable 2
        Output:
                u   (float)     wave amplitude
        '''
        u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)

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

class visual:
    def plot_all(sol, rangeX, rangeT):
        '''plot the full t-x KdV solution
        Input:
            sol         (array)
            rangeX      (float/int)
            rangeT      (float/int)
        Output:
            None    
        '''
        plt.figure(figsize=(8,8))
        plt.imshow(sol[::-1, :], extent=[0,rangeX,0,rangeT])
        #sol[::-1, :] to flip the direction of wave to positive x
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.axis('auto')
        plt.title('Korteweg-de Vries on a Periodic Domain')
        plt.show()

        return 

    def plot_profile(sol, t, N):
        '''plot the wave profile at time t in the periodic domain with range N
        Input:
            sol     (array)     the solution array of the wave equation
            t       (int)       time instance of the wave profile
            N       (int)       the number of spatial steps
            
        Output:
            plot of the wave profile at time t      (plt)'''
        for i in range(len(sol)):       #loop through every frame
            if i == t:
                fig, ax = plt.subplots()   #initialize figure
                ax.plot(np.arange(N), sol[t])
                ax.set_xlabel('Position x')
                ax.set_ylabel('Amplitude')
                ax.set_title('Wave Profile at Time ' + str())
                plt.show()

        return 

    def plot_video(sol, T, N, fps=20):
        '''construct a video of moving wave profile
        Input:
            sol     (array)    the solution array of the wave equation
            T       (int)      the number of time steps
            N       (int)      the number of spatial steps
            fps     (int)      time instance of the wave profile
            
        Output:
            plot of the wave profile at time t      (plt)'''
        images=[]       #set up image container
        maxAmp = max(sol[0])
        for i in range(T):      #loop through every frame
            if i%5 == 0:        #sample every 5 frames
                fig, ax = plt.subplots()   #initialize figure
                ax.plot(np.arange(N), sol[i])
                ax.set_xlabel('Position x')
                ax.set_ylabel('Amplitude')
                ax.set_ylim(0, maxAmp)
                ax.set_title('Wave Profile at Time ' + str())
                images = imv.memory.savebuff(plt, images)       #save image to the temp container
                plt.clf()
        imv.memory.construct(images, 'kdv_profile.avi', fps)        #construct the video 

        return

class numAnalysis:
    def amplitude():
        '''the numerical analysis of the wave amplitude over time'''

        return

    def coefficient():
        '''the numerical analysis of coefficients and stability'''

        return

if __name__ == "__main__":
    main()

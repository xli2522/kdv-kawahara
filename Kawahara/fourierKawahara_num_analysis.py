# Kawahara Numerical Stability Analysis
# Import libraries
import numpy as np
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
import matplotlib.pyplot as plt
import imvideo as imv       #dependent on opencv; use pip install opencv-python: note: it uses namespace cv2
import time
from tqdm import tqdm       #progress bar

def main():
    '''main numerical Kawahara solution function'''
    # Set the size of the domain, and create the discretized grid.
    L = 2*np.pi      #length of periodic boundary
    N = 27      #number of spatial steps
    dx = L / (N - 1.0)      #spatial step size
    x = np.linspace(0, (1-1.0/N)*L, N)      #initialize x spatial axis
    
    main_start = time.time()    

    # Set the time sample grid.
    T = 2
    t = np.linspace(0, T, 400)
    dt = len(t)
    
    '''
    ############# TEST STATIONARY SOLUTION U0 #############
    #############                             #############
    # guessed parameters
    param = [1, 0.25, 1, 0.01, -0.1798]      #[alpha, beta, sigma, epsilon, lamb]

    # Not exact for two solitons on a periodic domain, but close enough...
    #u0 = waveEquation.kdv_soliton_solution(x-0.33*L, 0.75)
    leading_terms = waveEquation.u0_leading_terms(param, 0.01)
    u0 = waveEquation.kawahara_stationary_solution(x, leading_terms, L, param)

    sol = waveEquation.solve_kawahara(waveEquation.kawahara_model, u0, t, L, param, 5000)
    print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))
    
    visual.plot_video(sol, len(t), N, 'kawahara_stationary_v0.avi')
    ############# TEST STATIONARY SOLUTION U0 #############
    #############                             #############
    '''
    #############     TEST FULL SOLUTION U    #############
    #############                             #############
    # guessed parameters
    param1 = [1, 0.25, 1, 0.01, -0.1798]      # [alpha, beta, sigma, epsilon, lamb]
    param2 = [0.01, -0.1798, 0.7845]                # [delta, lamb, mu]

    leading_terms_u0 = waveEquation.u0_leading_terms(param1, 0.01)
    stationary_u0 = waveEquation.kawahara_stationary_solution(x, leading_terms_u0, L, param1)
    leading_terms_u1 = waveEquation.u1_leading_terms()
    combined_u =  waveEquation.kawahara_combined_solution(stationary_u0, x, leading_terms_u1, param2, 0.01)
    sol = waveEquation.solve_kawahara(waveEquation.kawahara_model, combined_u, t, L, param1, 5000)
    print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))

    visual.plot_video(sol, len(t), N, 'kawahara_combined_v0_Ttest.avi')
    #############     TEST FULL SOLUTION U    #############
    #############                             #############

    #visual.plot_profile(sol, 250, N)
    #numAnalysis.amplitude(sol, len(t))
    #visual.plot_all(sol, L, T)

    return

class waveEquation:
    '''class waveEquation is under development and testing'''
    # The Kawahara model using Fast Fourier Transform
    def kawahara_model(u, t,  L, param, follow_wave_profile=False):
        '''The Kawahara model using FFT
        Input:
                u       (float)     wave amplitude
                t       (float)     simulation duration
                L       (float)     X length for periodic boundaries 
                param   (list)      list of parameters [alpha, beta, sigma, epsilon, lamb]
                follow_wave_profile       (boolean)     defaul = False
        Output:
                dudt    (float)     left side of the time differential
        '''
        alpha, beta, sigma, _, __,= param
        v0 = alpha - beta

        #Compute the space differentials of u
        ux = psdiff(u, period=L)                #1st order differential 
        uxxx = psdiff(u, period=L, order=3)     #3rd order differential 
        uxxxxx = psdiff(u, period=L, order=5)   #5th order differential
        u2x = psdiff(u**2, period=L)            #1st order differential; accuracy needs to be confirmed
        
        if follow_wave_profile:
            # moving with wave
            dudt = v0*ux + alpha*uxxx + beta*uxxxxx + sigma*u2x #Kawahara model; moving reference frame; time differential
        else:
            dudt = alpha*uxxx + beta*uxxxxx + sigma*u2x  

        return dudt

    def u0_leading_terms(param, a1):
        '''Calculates the leading terms in equation (24) - the stationary term u0
        Input:
                param               (list)      list of parameters [alpha, beta, sigma, epsilon, lamb]
                a1                  (float)     the initial guess of a1
        Output:
                leading_terms       (list)      list of leading terms in equation (24)
                                                    [a1, a0, a2, a3, a4]
         '''
        alpha, beta, sigma, epsilon, _, = param
        v0 = alpha - beta

        a0 = -(sigma/2)*(1/v0)*a1**2                                       #equation (28)
        a2 = -(sigma/2)*(1/(v0-4*alpha+16*beta))*a1**2                     #equation (29)
        a3 = -(sigma/2)*(1/(v0-9*alpha+81*beta))*2*a2*a1                   #equation (30)
        a4 = -(sigma/2)*(1/(v0-16*alpha+256*beta))*(a2**2+2*a2*a1)         #equation (31)
        
        leading_terms = [a1, a0, a2, a3, a4]

        return leading_terms

    def u1_leading_terms(param=None):
        '''Calculates the leading terms in equation (24) - the perturbation term u1
        Input:
                param               (list)      list of parameters [alpha, beta, sigma, epsilon, lamb]
                                    default = None
        Output:
                leading_terms       (list)      list of leading terms in equation (9)
                                                    [b1, b2, b3, b4]
        '''
        if param != None:
            alpha, beta, sigma, epsilon, lamb = param
        b1 = b2 = b3 = b4 = 0.001
        leading_terms = b1, b2, b3, b4 

        return leading_terms

    def kawahara_stationary_solution(x, leading_terms, L, param=None):
        '''The stable solution to the Kawahara
            reference: 
                STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS 
                    TO THE KAWAHARA EQUATION (Olga Trichtchenko et al.)
            refered to as u0; equation (24)
            elements obtained by solving (28), (29), (30), (31)
        Input:
                x                   (float)     position 
                leading_terms       (list)      leading terms = [a1, a0, a2, a3, a4]
                L                   (int)       length of the periodic range
                param               (list)      parameters = [alpha, beta, sigma, epsilon, lamb],
                                                                                default = None
        Output:
                u0                  (float)     amplitude
        NOTE: this stable solution has period (approximate) 2pi; under testing
        '''
        a1, a0, a2, a3, a4 = leading_terms

        #u0 = a0 + a1*np.cos(2*np.pi/L*x) + a2*np.cos(2*np.pi/L*2*x) + a3*np.cos(2*np.pi/L*4*x)
        u0 = a0 + a1*np.cos(x) + a2*np.cos(2*x) + a3*np.cos(4*x)
        return u0

    def kawahara_combined_solution(u0, x, leading_terms_u1, param, t):
        '''The combined solution to the Kawahara --> u0 stationary + u1 perturbation
            reference: 
                STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS 
                    TO THE KAWAHARA EQUATION (Olga Trichtchenko et al.)
            refered to as u1
        Input:
                initial amplitude of the stable solution
                u0                  (func)      kawahara_stable_solution ==> u0 in equation (5)
                x                   (float)     position
                param               (list)      [delta, lamb, mu]  eigenvalue
                leading_terms_u1    (list)      leading terms in equation (9)
                t                   (float)     time

        Output:
                u                   (float)     the unstable initial solution to the Kawahara
        Note:   lamb -->  the eigenvalue  
                    tentative values:   1) stable: -0.1798; 2) unstable: 7*10**(-6) + i0.277
        '''
        delta, lamb, mu = param
        b1, b2, b3, b4 = leading_terms_u1

        #experiment with complex calculation accuracy
        u1 = b1*np.exp(1j*(mu-1)*x) + b2*np.exp(1j*(mu-2)*x) + b3*np.exp(1j*(mu-3)*x) + b4*np.exp(1j*(mu-4)*x) + \
                b1*np.exp(1j*(mu+1)*x) + b2*np.exp(1j*(mu+2)*x) + b3*np.exp(1j*(mu+3)*x) + b4*np.exp(1j*(mu+4)*x)

        u = np.real(u0 + delta*np.exp(lamb*t)*u1)        # take the real part of the solution
        
        return u

    def kdv_soliton_solution(x, c):
        '''The approximated exact soliton solution to general KdV
        Input: 
                x   (float)     variable 1
                c   (float)     variable 2
        Output:
                u   (float)     wave amplitude
        '''
        u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)

        return u

    def solve_kawahara(model, u0, t, L, param, steps):
        '''Solve the Kawahara using Scipy odeint
        Input:
            model               Kawahara model
            u0      (array)     initial amplitude
            t       (array)     time range
            L       (int)       periodic range
            param   (list)      parameters = [alpha, beta, sigma, epsilon, lamb] 
            steps   (int)       maximum steps allowed
        Output:
            sol     (array)     periodic solutions'''

        sol = odeint(model, u0, t, args=(L, param, True), mxstep=steps)

        return sol

class visual:
    '''visual display of results'''
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
            plt.title('Kawahara on a Periodic Domain')
            plt.show()

            return 

    def plot_profile(sol, t, N):
        '''plot the wave profile at time t in the periodic domain with range N
        Input:
            sol     (array)     the solution array of the wave equation
            t       (int)       time instance of the wave profile
            N       (int)       the number of spatial steps
            
        Output:
            None'''

        for i in range(len(sol)):       #loop through every frame
            if i == t:
                fig, ax = plt.subplots()   #initialize figure
                ax.plot(2*np.pi/N*np.arange(N), sol[t])
                ax.set_xlabel('Position x')
                ax.set_ylabel('Amplitude')
                ax.set_title('Wave Profile at Time ' + str())
                plt.show()

        return 

    def plot_video(sol, T, N, title, fps=20):
        '''construct a video of moving wave profile
        Input:
            sol     (array)    the solution array of the wave equation
            T       (int)      the number of time steps
            N       (int)      the number of spatial steps
            fps     (int)      time instance of the wave profile
            title   (string)   video file title; default = None* see BUG
            
        Output:
            plot of the wave profile at time t      (plt)

        BUG: setting title=None and then define title to a string 
                            causes imvideo --> cv2 --> unable to obtain image spec error
        '''
        images=[]       #set up image container
        maxAmp = max(sol[0])
        for i in tqdm(range(T)):      #loop through every frame
            if i%2 == 0:        #sample every 2 frames
                fig, ax = plt.subplots()   #initialize figure
                ax.plot(2*np.pi/N*np.arange(N), sol[i])
                ax.set_xlabel('Position x')
                ax.set_ylabel('Amplitude')
                ax.set_ylim(-1.1*maxAmp, 1.1*maxAmp)
                ax.set_title('Wave Profile at Time ' + str())
                images = imv.memory.savebuff(plt, images)       #save image to the temp container
                plt.close()

        imv.memory.construct(images, str(title), fps)        #construct the video 

        return

class numAnalysis:
    '''numerical stability analysis'''
    def amplitude(sol, T):
        '''the numerical analysis of the wave amplitude over time
        Input: 
            sol     (array)     the solution array
            T       (int)       the number of time steps
        Output:
            plot of change in amplitude of the wave     (plt)'''
        amp = np.zeros(len(sol))

        for i in range(len(sol)):
            amp[i]=max(sol[i])
        
        fig, ax = plt.subplots()
        ax.scatter(range(T), amp)
        ax.set_xlim(0, T)
        ax.set_ylim(max(0.2*sol[0]), 1.5*max(sol[0]))
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Amplitude')
        ax.set_title('Amplitude vs. Time Step')

        plt.show()

        return

    def stable(sol):
        '''check if the solution is long-term stable based on a set of criteria
        Input: 
                sol     (array)     solution array
        Output:
                ?

        Criteria for stable candidate:
        1)  integration speed is greater then x steps per second
        2)  calculated wave profile is smooth (second derivative within a range [a, b])
        3)  no sudden change of wave profile between immediate time steps
        sample results every 5 time steps --> in each sampled time step, sample space steps around the peak
        '''

        return

    def coefficient():
        '''the numerical analysis of coefficients and stability'''

        return

class paraEst:
    '''numerical parameter estimation'''
    def stationary_param(stable):
        '''tests parameters in the stationary solution u0 ranges from [0.? to 0.?] based on experiments + guesses
        reference: 
                STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS 
                    TO THE KAWAHARA EQUATION (Olga Trichtchenko et al.)
            refered to as u0; equation (24)
            elements obtained by solving (28), (29), (30), (31)
        Input:
                stable          (function)          the stability testing function from numAnalysis
        Output:
                ?

        Criteria for stable candidate:
        1)  integration speed is greater then x steps per second
        2)  calculated wave profile is smooth (second derivative within a range [a, b])
        3)  no sudden change of wave profile between immediate time steps
        sample results every 5 time steps --> in each sampled time step, sample space steps around the peak
        '''

        return 

    def perturbation_param():
        '''tests parameters in the stationary solution u1 ranges from [0.? to 0.?] based on experiments + guesses
        reference: 
                STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS 
                    TO THE KAWAHARA EQUATION (Olga Trichtchenko et al.)
            refered to as u1; parameters see equation (9)

        Criteria for stable candidate:
        1)  integration speed is greater then x steps per second
        2)  calculated wave profile is smooth (second derivative within a range [c, d])
        3)  no sudden change of wave profile between immediate time steps
        sample results every 5 time steps --> in each sampled time step, sample space steps around the peak
        '''

        return
    
    def kawahara_param():

        return 

###  $$$    NOTE    $$$    ###
#may use double for higher precision when necessary

###  $$$    BUG    $$$    ###
#setting title=None and then define title to a string 
#      causes imvideo --> cv2 --> unable to obtain image spec error

if __name__ == "__main__":
    main()
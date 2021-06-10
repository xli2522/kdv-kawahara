# Kawahara Numerical Stability Analysis
# Import libraries
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.function_base import average
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
    N = 30      #number of spatial steps
    dx = L / (N - 1.0)      #spatial step size
    x = np.linspace(0, (1-1.0/N)*L, N)      #initialize x spatial axis    

    # Set the time sample grid.
    T = 20
    t = np.linspace(0, T, 4000)
    dt = len(t)
    
    '''
    #############     TEST FULL SOLUTION U    #############
    #############                             #############
    # guessed parameters
    mus = [0.7845, 0.6324, -0.7928]
    lambs = [-0.1798, 0.2277, 0.2128]
    std = np.zeros(len(mus))
    for i in tqdm(range(len(mus))):
        main_start = time.time()

        param1 = [1, 0.25, 1, 0.01, lambs[i] ]      # [alpha, beta, sigma, epsilon, lamb]
        param2 = [0.01, lambs[i], mus[i], 1, 0.25, 1]                # [delta, lamb, mu, alpha, beta, sigma]
        a1 = param1[3]      #a1 = epsilon
        leading_terms_u0 = waveEquation.u0_leading_terms(param1, a1)
        #with open('kawahara_parameters.txt', "w") as f:
                #f.write(str(main_start)+' '+ str(T)+' times '+ str(len(t))+' steps main parameters: '+str(param1)+' u0 parameters: '+ str(leading_terms_u0))
                
        stationary_u0 = waveEquation.kawahara_stationary_solution(x, leading_terms_u0, L, param1)
        leading_terms_u1 = waveEquation.u1_leading_terms(param2)
        combined_u =  waveEquation.kawahara_combined_solution(stationary_u0, x, leading_terms_u1, param2, 0.01)     # t = 0.01
        sol = waveEquation.solve_kawahara(waveEquation.kawahara_model, combined_u, t, L, param1, 5000)
        print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))

        std[i] = numAnalysis.amplitude(sol, len(t), title='Table1_'+str(i)+'_a1_'+str(a1)+'.png')
        visual.plot_video(sol, len(t), N, 'Table1_'+str(i)+'_a1_'+str(a1)+'.avi')

    numAnalysis.simple_plot(std)

    # SAVE THE FULL SOLUTION
    #with open('kawahara_'+str(T)+'time'+str(len(t))+'steps'+'_beta_'+str(param1[1])+str(main_start)+'.txt', "w") as f:
        #for i in sol:
            #f.write(str(i))

    #############     TEST FULL SOLUTION U    #############
    #############                             #############'''

    #############     Table 2   SOLUTION U    #############
    #############                             #############
   
    mus = [5.09,5.48,4.79,5.02,4.16,4.23,3.24,-3.23,2.27,2.36,1.49,1.64,0.74,0.69]
    lambs = [62.82,51.62,35.98,19.78,7.09,0.595,-0.219,-0.305,-3.22,-13.41,-29.71,-49.13,-64.87,-57.60]
    std = np.zeros(len(mus))

    for i in tqdm(range(len(mus))):
        param1 = [1, 3/160, 1, 0.01, lambs[i]]      # [alpha, beta, sigma, epsilon, lamb]
        param2 = [0.01, lambs[i], mus[i], 1, 3/160, 1]                # [delta, lamb, mu, alpha, beta, sigma]
        a1 = param1[3]
        leading_terms_u0 = waveEquation.u0_leading_terms(param1, a1)
        stationary_u0 = waveEquation.kawahara_stationary_solution(x, leading_terms_u0, L, param1)
        leading_terms_u1 = waveEquation.u1_leading_terms(param2)
        combined_u =  waveEquation.kawahara_combined_solution(stationary_u0, x, leading_terms_u1, param2, 0.01)
        sol = waveEquation.solve_kawahara(waveEquation.kawahara_model, combined_u, t, L, param1, 5000)
        # SAVE THE FULL SOLUTION
        with open('kawahara_Table2_'+str(i)+'_'+str(T)+'time'+str(len(t))+'steps'+'.txt', "w") as f:
            for row in sol:
                f.write(str(row))

        std[i] = numAnalysis.amplitude(sol, len(t), title='Table2_2pi_'+str(i)+'.png')
        visual.plot_video(sol, len(t), N, 'Table2_2pi_'+str(i)+ '.avi')
        #visual.plot_profile(sol, 250, N)
        #visual.plot_all(sol, L, T)

    numAnalysis.simple_plot(std)    

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
        if float(beta) == 0.2:
            a2 = a1/2
        else:
            a2 = -(sigma/2)*(1/(v0-4*alpha+16*beta))*a1**2                     #equation (29)
        a3 = -(sigma/2)*(1/(v0-9*alpha+81*beta))*2*a2*a1                   #equation (30)
        a4 = -(sigma/2)*(1/(v0-16*alpha+256*beta))*(a2**2+2*a2*a1)         #equation (31)
        leading_terms = [a1, a0, a2, a3, a4]

        return leading_terms

    def u1_leading_terms(param=None):
        '''Calculates the leading terms in equation (24) - the perturbation term u1
            The calculation follows the procedure described in equation (35), (36), (37), (38)
        Input:
                param               (list)      list of parameters [delta, lamb, mu, alpha, beta, sigma]
                                    default = None
        Output:
                leading_terms       (list)      list of leading terms in equation (9)
                                                    [b1, b2, b3, b4]
        '''
        if param != None:
            delta, lamb, mu, alpha, beta, sigma = param
        V = alpha - beta

        #MATRIX OPERATION
        '''
        #S = iD + iT
        #Matrix D
        D = np.zeros(8)     # only take the first 8 leading terms
        #Matrix T
        matrixT = np.zeros((8, 8))
        for i in range(len(D)):
            n = i - len(D)/2       #len(D) is guaranteed to be even
            D[i] = (n + mu)*V - (-n + mu)**3*alpha + (n + mu)**5*beta

            for n in range(len(matrixT[0])):
                matrixT[i][n] = 2*sigma
        matrixD = np.diag(D)

        matrixS = 1j*matrixD*1j*matrixT
        '''
        b1 = b2 = b3 = b4 = 0.01
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
                param               (list)      [delta, lamb, mu, alpha, beta, sigma]  eigenvalue
                leading_terms_u1    (list)      leading terms in equation (9)
                t                   (float)     time

        Output:
                u                   (float)     the unstable initial solution to the Kawahara
        Note:   lamb -->  the eigenvalue  
                    tentative values:   1) stable: -0.1798; 2) unstable: 7*10**(-6) + i0.277
        '''
        delta, lamb, mu, _, __, ___,= param
        b1, b2, b3, b4 = leading_terms_u1

        #experiment with complex calculation accuracy
        u1 = b1*np.exp(1j*(mu-1)*x) + b2*np.exp(1j*(mu-2)*x) + b3*np.exp(1j*(mu-3)*x) + b4*np.exp(1j*(mu-4)*x) + \
                b1*np.exp(1j*(mu+1)*x) + b2*np.exp(1j*(mu+2)*x) + b3*np.exp(1j*(mu+3)*x) + b4*np.exp(1j*(mu+4)*x)

        u = np.real(u0 + delta*np.exp(lamb*t)*u1)        # take the real part of the solution
        #u = np.real(delta*np.exp(lamb*t)*u1)

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

    def plot_profile(sol, t, N, title):
        '''plot the wave profile at time t in the periodic domain with range N
        Input:
            sol     (array)     the solution array of the wave equation
            t       (int)       time instance of the wave profile
            N       (int)       the number of spatial steps
            title   (string)    title
        Output:
            None'''

        for i in range(len(sol)):       #loop through every frame
            if i == t:
                fig, ax = plt.subplots()   #initialize figure
                ax.plot(2*np.pi/N*np.arange(N), sol[t])
                ax.set_xlabel('Position x')
                ax.set_ylabel('Amplitude')
                ax.set_title('Wave Profile at Time ' + str(title))
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
        minAmp = min(sol[0])
        for i in tqdm(range(T)):      #loop through every frame
            if i%5 == 0:        #sample every 5 frames
                fig, ax = plt.subplots()   #initialize figure
                ax.plot(2*np.pi/N*np.arange(N), sol[i])
                ax.set_xlabel('Position x')
                ax.set_ylabel('Amplitude')
                ax.set_ylim(1.1*minAmp, 1.1*maxAmp)
                ax.set_title('Wave Profile at Time ' + str())
                images = imv.memory.savebuff(plt, images)       #save image to the temp container
                plt.close()

        imv.memory.construct(images, str(title), fps)        #construct the video 

        return

class numAnalysis:
    '''numerical stability analysis'''
    def amplitude(sol, T, title=None, mid=None):
        '''the numerical analysis of the max wave amplitude over time
        Input: 
            sol     (array)     the solution array
            T       (int)       the number of time steps
            title   (string)    default = None
        Output:
            ampStd  (float)     standard deviation
            plot of change in amplitude of the wave     (plt)'''
        amp = np.zeros(len(sol))

        for i in range(len(sol)):
            value = abs(max(sol[i])-min(sol[i]))
            amp[i] = value

        ampStd = np.std(amp)
        fig, ax = plt.subplots()
        ax.scatter(range(T), amp, s=40)
        ax.set_xlim(0, T)
        ax.set_ylim(average(amp) - max(amp)/16, average(amp) + max(amp)/16)
        ax.set_xlabel('Time Step     STD: '+str(ampStd))
        ax.set_ylabel('Amplitude Difference')
        ax.set_title('Amplitude Difference vs. Time Step')

        if title == None:
            plt.show()
        else:
            plt.savefig(str(title))
        plt.clf()

        return ampStd

    def simple_plot(dat, id=None):
        '''Simple Plot'''
        fig, ax = plt.subplots()   #initialize figure
        ax.plot(np.arange(len(dat)), dat)
        ax.set_xlabel('Instance')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Standard Deviation vs. Instance')
        plt.savefig('Std_'+str(time.time())+'_'+str(id)+'.png')
        
        return

class operation:
    def num_continue():
        '''Resume a numerical calculation of the Kawahara'''

        return

    def num_batches(n):
        '''Numerical calculation of the Kawahara in n batches'''
        T_batch = 2
        print('batch progress...')
        for i in tqdm(range(n)):
            T_batch += i 
            t_batch = np.linspace(T_batch - 2, T_batch, 400)

        return

###  $$$    NOTE    $$$    ###
#may use double for higher precision when necessary

###  $$$    BUG    $$$    ###
#setting title=None and then define title to a string 
#      causes imvideo --> cv2 --> unable to obtain image spec error

if __name__ == "__main__":
    main()
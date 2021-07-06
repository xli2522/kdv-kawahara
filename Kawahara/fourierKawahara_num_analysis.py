# Kawahara Numerical Stability Analysis - X. Li 2021, Western University
# Import libraries
import numpy as np
from numpy.core.fromnumeric import size
from numpy.lib.function_base import average
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
from scipy.optimize import fsolve
import pandas as pd         #data logging
import matplotlib.pyplot as plt
import imvideo as imv       #dependent on opencv; use pip install opencv-python: note: it uses namespace cv2
import time
from tqdm import tqdm       #progress bar

###  $$$    NOTE    $$$    ###
# (1) u0 and u1 leading term selection order needs to be verified
# (2) add damping coefficient to function

###  $$$    BUG    $$$    ###
# (1) setting title=None and then define title to a string 
#      causes imvideo --> cv2 --> unable to obtain image spec error

def test():
    '''Full numerical Kawahara solution test function
    Tentative:
        Group 1: Analytical Instability
            Test 1: Floquet Parameter Test
        Group 2: Numerical Instability
            Test 2: Kawahaa Numerical Solver (stable condition)
            Test 3: Kawahara Numerical Solver (unstable condition)
    '''
    #############     Floquet Parameter mu    #############
    #############                             #############
    #param_damping = [1, 0.25, 1, 0.01]      # [alpha, beta, sigma, gamma]     include gamma for damping
    #param_no_damping = [1, 0.25, 1]
    #lamb = analytical.lambdaFloquet(0.62, 0.66, 400, param_damping, savePic=True, damping=True)
    #lamb = analytical.lambdaFloquet(0.62, 0.66, 400, param_no_damping, savePic=True, damping=False)
    #############                             #############
    #############     Floquet Parameter mu    #############
    

    #############     Numerical Instability   #############
    #############                             #############
    #mus = [5.09,5.48,4.79,5.02,4.16,4.23,3.24,-3.23,2.27,2.36,1.49,1.64,0.74,0.69]
    #lambs = [62.82,51.62,35.98,19.78,7.09,0.595,-0.219,-0.305,-3.22,-13.41,-29.71,-49.13,-64.87,-57.60]
    mus = [0.74, 4.79]             
    lambs = [-64.87, 35.98]  
    #mus = [5.09]             
    #lambs = [62.82]   
    beta = 3/160
    #beta = 1/4    
    #mus = [0.7845, 0.6324, -0.7928]
    #lambs = [-0.1798, 0.2277, 0.2128]  
    damp_all_cases = False

    avg = np.zeros(len(mus))       
    main_start = time.time()

    for i in tqdm(range(len(mus))):
        print('\n')                                           #prevent tqdm from consuming the first printed character in terminal

        #####################################################################
        # Set the size of the domain, and create the discretized grid.
        #L = 2*np.pi/(abs(mus[i])-np.floor(abs(mus[i])))      #length of periodic boundary
        L = 240*np.pi
        #force a larger periodic domain
        if L >= 10*np.pi:
            L = L
        else:
            L = 10*np.pi
        
        N = int(np.floor(30*L/(2*np.pi)))       #number of spatial steps; fit to the length of the periodic domain
        dx = L / (N - 1.0)                      #spatial step size
        x = np.linspace(0, (1-1.0/N)*L, N)      #initialize x spatial axis    
        # Set the time sample grid.
        T = 1
        t = np.linspace(0, T, 800)
        dt = len(t)

        ######################################################################
        param_damping = [1, beta, 1, 0.01]
        param_no_damping = [1, beta, 1]
        #calculate Re{lambda} values in the Floquet-Re{lambda} space
        lamb_original = analytical.lambdaFloquet(0.97*mus[i], 1.03*mus[i], 800,
                                        param_no_damping, savePic=True, plot=False, damping=False)        
        if damp_all_cases:
            #this is incorrect atm; find_stable_lamb outputs the Re{lambda} instead of the Im{lambda}
            lamb = analytical.find_stable_lamb(param_damping, mus[i], 800, savePic=True, plot=False)
            mu = mus[i]

        else:
            lamb = lambs[i]
            mu = mus[i]

        ######################################################################
        param1 = [1, beta, 1, 0.001, lamb]                              # [alpha, beta, sigma, epsilon, lamb]
        param2 = [0.0005, lamb, mu, 1, beta, 1]                         # [delta, lamb, mu, alpha, beta, sigma]
        a1 = 0.001   #param1[3]                                          #DECREASE A1 WHEN BETA IS LARGE
 
        kModes=10                           #number of fourier modes to consider
        ic_start = time.time()
        
        ###########################     U0  aN    ###########################
        #approximate the optimized u0 coeficients for analytical approximation of perturbation solution u1
        optimized_u0, v0, stationary_u0 = analytical.optimize_u0Coeff(param_no_damping, a1, a1, 
                            steps=1, L=L, spaceResolution=N, plot=False, N=2*kModes+1, damping=damp_all_cases)
    
        ###########################     U0  wE    ###########################
        #calculate the leading Fourier coefficients of the stationary solution u0 class waveEquation method
        #u0_leading_coeff = waveEquation.u0_leading_coeff(param1, a1, v0)
        
        #calculate the stationary solution u0 itself using u0_leading_coeff
        #stationary_u0 = waveEquation.kawahara_stationary_solution(x, u0_leading_coeff, L, param1)
        
        ###########################     U1  aN    ###########################
        #calculate the leading terms of the perturbation solution u1
        lambdaCalc, U1 = analytical.fourierCoeffMatrix(param_no_damping, optimized_u0, v0, mu, damping=damp_all_cases)
        
        #calculate the perturbation solution u1
        perturbation_u1 = analytical.collect_u1(lambdaCalc, U1, mus[i], x)
        
        ###########################     Uc  wE    ###########################
        #calculate the combined solution u
        combined_u =  waveEquation.kawahara_combined_solution(stationary_u0, x, param2, 0, u1=perturbation_u1)

        ###########################     Solve     ###########################
        print("Initial condition calculation --- %s seconds ---" % (time.time() - ic_start))
        solver_start = time.time()
        sol = waveEquation.solve_kawahara(waveEquation.kawahara_model, combined_u, t, L, param1, 5000, modelArg=(True, False, v0))
        print("Numerical solver --- %s seconds ---" % (time.time() - solver_start))
        print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))
        
        # SAVE THE FULL SOLUTION
        with open('Table_test_'+str(int(L/np.pi))+'pi_'+str(mus[i])+'mu_'+str(i)+'_'+str(T)+'time'+str(len(t))+'steps'+'.txt', "w") as f:
            for row in sol:
                f.write(str(row))

        avg[i] = numAnalysis.amplitude(sol, len(t), title='Table_test_'+str(a1)+'maxamp_'+str(mus[i])+'_'+str(int(L/np.pi))+'pi_'+str(i)+'.png')
        #visual.plot_video(sol, len(t), N, L, 'Table_test_'+str(int(L/np.pi))+'pi_'+str(mus[i])+'mu_'+str(i)+ '.avi', fps=30)
        
        #visual.plot_profile(sol, np.rint(3*dt/4), N)
        #visual.plot_all(sol, L, T, 'Table_test_Full_Profile_'+str(int(L/np.pi))+'pi_'+str(mus[i])+'mu_'+str(i)+ '.png')
    #numAnalysis.simple_plot(avg)
        #############                             #############
        #############     Numerical Instability   #############'''

    return

class waveEquation:
    '''Numerical wave equation stability related functions'''

    def kawahara_model(u, t,  L, param, follow_wave_profile=True, damping=False, v0=None):
        '''The Kawahara model
        Input:
                u                       (float)             wave amplitude
                L                       (float)             X length for periodic boundaries 
                param                   (list)              list of parameters [alpha, beta, sigma, epsilon, lamb]
                follow_wave_profile     (boolean)           moving reference frame with travelling wave
                damping                 (boolean)           activate damping term gamma*uxx (beta)
        Output:
                dudt                    (float)             left side of the time differential
        Details:
                diff method             spectral derivative
                damping                 beta feature
                reference               STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                            OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                                arXiv:1806.08445v1
                                                    21 Jun 2018
        '''
        
        alpha, beta, sigma, epsilon, __,= param
        if v0 == None:                              #use approximated v0 if analytical v0 is not provided
            v0 = alpha - beta
        if damping:                                 #avoid uxx calculation if no damping
            gamma = 0.1*epsilon                     #damping coefficient - 0.043 boundary (previous version) - current gamma = a1
            uxx = psdiff(u, period=L, order=2)      #2nd order differential

        #compute the spectral derivatives of u
        ux = psdiff(u, period=L)                    #1st order differential 
        uxxx = psdiff(u, period=L, order=3)         #3rd order differential 
        uxxxxx = psdiff(u, period=L, order=5)       #5th order differential
        u2x = psdiff(u**2, period=L)                #1st order differential; accuracy needs to be confirmed
        
        if follow_wave_profile:                     #moving reference frame
            if damping:                             #avoid uxx calculation if no damping
                #kawahara model with damping
                dudt = v0*ux + alpha*uxxx + beta*uxxxxx + sigma*u2x + gamma*uxx
            else:
                #kawahara model - equation (4)
                dudt = v0*ux + alpha*uxxx + beta*uxxxxx + sigma*u2x
        else:
            #kawahara model - equation (1)
            dudt = alpha*uxxx + beta*uxxxxx + sigma*u2x         

        return dudt

    def u0_leading_coeff(param, a1, v0=None):
        '''Approximate the first 4 leading terms of the stationary/stable solution u0
        Input:
                param                       (list)              list of parameters [alpha, beta, sigma, epsilon, lamb]
                a1                          (float)             the initial value/guess of a1
                v0                          (float)             the wave speed
        Output:
                leading_terms               (list)              list of 4 leading terms in equation (24)
                                                                    [a1, a0, a2, a3, a4]
        Details: 
                Not Used                    Not used due to low accuracy (theoretically; minimal difference in practice 
                                                compared to class analytical func collect_u0)
                u0 stable solution          equation (24) 
                reference                   STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                                OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                                    arXiv:1806.08445v1
                                                        21 Jun 2018
         '''
        alpha, beta, sigma, _, __, = param
        if v0 == None:                                                     #use approximated v0 if analytical v0 is not provided
            v0 = alpha - beta

        a0 = -(sigma/2)*(1/v0)*a1**2                                       #equation (28)
        if float(beta) == 0.2:                                             #approximate a2 in case of 0 division
            a2 = a1/2
        else:
            a2 = -(sigma/2)*(1/(v0-4*alpha+16*beta))*a1**2                 #equation (29)
        a3 = -(sigma/2)*(1/(v0-9*alpha+81*beta))*2*a2*a1                   #equation (30)
        a4 = -(sigma/2)*(1/(v0-16*alpha+256*beta))*(a2**2+2*a2*a1)         #equation (31)

        leading_terms = [a1, a0, a2, a3, a4]

        return leading_terms

    def fourierCoeffMatrix(param, leadingu0, v0=None):
        '''Calculate the Fourier coefficient eigenvector for the perturbation/unstable solution u1
        Input:
                param                       (list)              list of parameters [alpha, beta, sigma, epsilon, lamb]
                leadingu0                   (list)              leading terms of the stable solution u0 - equation (24)
        Output:
                lambdaCalc                  (array)             the eigenvalue of matrix S - equation (35)
                U1                          (array)             the eigenvector of matrix S - equation (35)
        Details: 
                Not Used                    Not used due to low accuracy (due to low kModes limited by class waveEquation func u0_leading_terms)
                u1 perturbation solution    equation (9) 
                reference                   STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                                OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                                    arXiv:1806.08445v1
                                                        21 Jun 2018
        '''
        _, __, mu, alpha, beta, sigma = param
        if v0 == None:                  #use approximated v0 if analytical v0 is not provided
            V = alpha - beta

        #MATRIX OPERATION
        kModes = 4                      #numerical approximation of kModes is limited to 4 
                                        #see class waveEquation: func u0_leading_terms(param, a1)
        matrixD = np.zeros((2*kModes+1, 2*kModes+1), 'complex')                 #matrixD - equation (36)
        matrixT = np.zeros((2*kModes+1, 2*kModes+1), 'complex')                 #matrixT - equation (36)
       
        for m in range(2*kModes+1):     
            for n in range(2*kModes+1):  
                ns = n - kModes   
                ms = m - kModes
                if ns == ms:
                    matrixD[m, n] = (ns + mu)*V - (ns + mu)**3*alpha + (ns + mu)**5*beta            #equivalent to equation (37)   
                else:
                    matrixT[m, n] = 2*sigma*(mu+ms)*leadingu0[abs(ns-ms)]                           #equivalent to equation (38)

        matrixS = 1j*matrixD + 1j*matrixT                                                           #equation (36)
        lambdaCalc, U1 = np.linalg.eig(matrixS)                                                     #equation (35)
        
        return lambdaCalc, U1

    def u1_leading_terms(param=None, leadingu0=None):
        '''Extract the first 4 leading terms of the perturbation solution u1 - equation (9)
        Input:
                param                       (list)              list of parameters [delta, lamb, mu, alpha, beta, sigma]
                leadingu0                   (list)              list of u0 leading terms                  
        Output:
                leading_terms               (list)              list of leading terms in equation (9)
                                                                            [b1, b2, b3, b4]
        Details: 
                Not Used                    Not used due to low accuracy (due to low kModes limited by class waveEquation func u0_leading_terms)
                about param=None            param=None triggers all bn = 0.00001; use with a1 = 0.000001 and class waveEquation funcs to explore interesting wave profiles
                u1 perturbation solution    equation (9) 
                reference                   STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                                OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                                    arXiv:1806.08445v1
                                                        21 Jun 2018
        '''
        if param != None:                       #extract result if param is given
            lambdaCalc, U1 = waveEquation.fourierCoeffMatrix(param, leadingu0)
            lambdaCalcMax = max(np.real(lambdaCalc))
            indVec = np.argwhere(np.real(lambdaCalc)==lambdaCalcMax)
            U1 = U1[:,indVec].transpose()
            
            b1, b2, b3, b4 = float(np.real(U1[0][0][0])), float(np.real(U1[0][0][1])), float(np.real(U1[0][0][2])), float(np.real(U1[0][0][3]))
            leading_terms = b1, b2, b3, b4
            
        else:                                   #approximate result if param not given
            b1 = b2 = b3 = b4 = 0.00001         #approximate leading bs
            leading_terms = b1, b2, b3, b4

        return leading_terms

    def kawahara_stationary_solution(x, leading_terms, L, param=None):
        '''The stable solution u0 to the Kawahara equation - equation (24)
            reference: 
                STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS 
                    TO THE KAWAHARA EQUATION (Olga Trichtchenko et al.)
            refered to as u0; equation (24)
            elements obtained by solving (28), (29), (30), (31)
        Input:
                x                               (float)             position x
                leading_terms                   (list)              leading terms = [a1, a0, a2, a3, a4, ...]
                L                               (int)               length of the periodic domain
                param                           (list)              parameters [alpha, beta, sigma, epsilon, lamb]
        Output:
                u0                              (float)             amplitude
        Details: 
                u0 stationary solution          equation (24) 
                reference                       STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                                    OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                                        arXiv:1806.08445v1
                                                            21 Jun 2018
        '''
        a1, a0, a2, a3, a4 = leading_terms[:5]          #select the first 5 leading terms

        u0 = a0 + a1*np.cos(2*np.pi/L*x) + a2*np.cos(2*np.pi/L*2*x) + a3*np.cos(2*np.pi/L*3*x)          #- equation (24)
        #u0 = a0 + a1*np.cos(x) + a2*np.cos(2*x) + a3*np.cos(3*x)

        return u0

    def kawahara_combined_solution(u0, x, param, t, leading_terms_u1=None, u1=None):
        '''The combined solution to the Kawahara --> u0 stationary + u1 perturbation
        Input:
                u0                              (func)              kawahara_stable_solution ==> u0 in equation (5)
                x                               (float)             position x
                param                           (list)              parameters [delta, lamb, mu, alpha, beta, sigma]
                t                               (float)             time of the initial condition
                leading_terms_u1                (list)              leading Fourier coefficients in equation (9)
                u1                              (float)             calculated final u1 result
        Output:
                u                               (float)             the combined aprroximated solution to the Kawahara
        Details: 
                leading_terms_u1 & u1           use either, never both
                u0 stationary solution          equation (24) 
                u1 perturbation solution        equation (9)
                u combined solution             equation (5)
                reference                       STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                                    OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                                        arXiv:1806.08445v1
                                                            21 Jun 2018
        '''
        delta, lamb, mu, _, __, ___,= param
        
        '''
        #include more terms in u1
        u1 = b1*np.exp(1j*(mu-1)*x) + b2*np.exp(1j*(mu-2)*x) + b3*np.exp(1j*(mu-3)*x) + b4*np.exp(1j*(mu-4)*x) + \
                b5*np.exp(1j*(mu-5)*x) + b6*np.exp(1j*(mu-6)*x) + b7*np.exp(1j*(mu-7)*x) + \
                b1*np.exp(1j*(mu+1)*x) + b2*np.exp(1j*(mu+2)*x) + b3*np.exp(1j*(mu+3)*x) + b4*np.exp(1j*(mu+4)*x) + \
                b5*np.exp(1j*(mu+5)*x) + b6*np.exp(1j*(mu+6)*x) + b7*np.exp(1j*(mu+7)*x) 
        '''
        if leading_terms_u1 != None:
            b1, b2, b3, b4 = leading_terms_u1
            u1 = b1*np.exp(1j*(mu-1)*x) + b2*np.exp(1j*(mu-2)*x) + b3*np.exp(1j*(mu-3)*x) + b4*np.exp(1j*(mu-4)*x) + \
                    b1*np.exp(1j*(mu+1)*x) + b2*np.exp(1j*(mu+2)*x) + b3*np.exp(1j*(mu+3)*x) + b4*np.exp(1j*(mu+4)*x) 

        u = np.real(u0 + delta*np.exp(1j*lamb*t)*u1)                               #equation (5)

        return u

    def kdv_soliton_solution(x, c):
        '''The approximated exact soliton solution to a general KdV
        Input: 
                x               (float)             variable 1
                c               (float)             variable 2
        Output:
                u               (float)             wave amplitude
        '''
        u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)

        return u

    def solve_kawahara(model, u0, t, L, param, steps, modelArg):
        '''Solve the Kawahara with Scipy odeint
        Input:
            model                                               Kawahara model
            u0                          (array)                 initial amplitude
            t                           (array)                 time range
            L                           (int)                   periodic range
            param                       (list)                  parameters [alpha, beta, sigma, epsilon, lamb] 
            steps                       (int)                   maximum steps allowed
            modelArg                    (list)                  Kawahara model arguments
        Output:
            sol                         (array)                 solutions
        '''
        sol = odeint(model, u0, t, args=(L, param, modelArg[0], modelArg[1], modelArg[2]), mxstep=steps)
        
        return sol

class visual:
    '''Visual display of results'''

    def plot_all(sol, rangeX, rangeT, title=None):
            '''Plot the full t-x solution on a periodic domain
            Input:
                    sol                 (array)             solution 
                    rangeX              (float/int)         range x
                    rangeT              (float/int)         range y
                    title               (str)               title
            '''
            plt.figure(figsize=(8,8))
            plt.imshow(sol[::-1, :], extent=[0,rangeX,0,rangeT])
            #sol[::-1, :] to flip the direction of wave to positive x
            plt.colorbar()
            plt.xlabel('x')
            plt.ylabel('t')
            plt.axis('auto')
            plt.title('Kawahara on a Periodic Domain')
            if title == None:
                plt.savefig('full_profile_'+str(time.time())+'_'+str(id)+'.png')
            else: 
                plt.savefig(str(title))

            return 

    def plot_profile(sol, t, N, inspect=True):
        '''plot the wave profile at time t in the periodic domain with range N
        Input:
                sol                 (array)                 the solution array of the wave equation
                t                   (int)                   time instance of the wave profile
                N                   (int)                   periodic domian length
                title               (string)                title
                inspect             (boolean)               display
        '''

        fig, ax = plt.subplots()   #initialize figure
        ax.plot(2*np.pi/N*np.arange(N), sol[int(t)])
        ax.set_xlabel('Position x')
        ax.set_ylabel('Amplitude')
        ax.set_title('Wave Profile at Time Step ' + str(t))
        if inspect:
            plt.show()

        return 

    def plot_video(sol, T, N, L, title, fps=20):
        '''construct a video of moving wave profile
        Input:
                sol                 (array)                 the solution array of the wave equation
                T                   (int)                   the number of time steps
                N                   (int)                   the number of spatial steps
                L                   (float)                 the length of the periodical domain
                title               (string)                video file title; default = None* see BUG
                fps                 (int)                   time instance of the wave profile
        BUG: 
                setting title=None and then define title to a string 
                                    causes imvideo --> cv2 --> unable to obtain image spec error
        '''
        images=[]                               #set up image container
        maxAmp = max(sol[0])
        minAmp = min(sol[0])
        for i in tqdm(range(T)):                #loop through every frame
            if i%2 == 0:                        #sample every 2 frames
                if T <= 4000:
                    fig, ax = plt.subplots()    #initialize figure
                    ax.plot(L/N/np.pi*np.arange(N), sol[i])
                    ax.set_xlabel('Position x (*pi)')
                    ax.set_ylabel('Amplitude')
                    ax.set_ylim(1.3*minAmp, 1.1*maxAmp)
                    ax.set_title('Wave Profile at Time Step: ' + str(i))
                    images = imv.memory.savebuff(plt, images)               #save image to the temp container
                    plt.close()
                else:
                    print('Total number of frames - ' + str(T))
                    batch = int(np.ceil(T/4000))
                    print('Devided in ' + str(batch) + 'batches.')

        imv.memory.construct(images, str(title), fps, inspect=False)        #construct the video 

        return

class numAnalysis:
    '''numerical stability analysis'''

    def amplitude(sol, T, title, inspect=False, savePic=True):
        '''Plot the max wave amplitude over time
        Input: 
                sol             (array)             the solution array
                T               (int)               the number of time steps
                title           (string)            title
                inspect         (noolean)           display
        Output:
                ampStd          (float)             standard deviation
        '''
        amp = np.zeros(len(sol))

        for i in range(len(sol)):
            value = abs(max(sol[i]))    
            amp[i] = value

        ampAvg = np.average(amp)
        fig, ax = plt.subplots()
        ax.scatter(range(T), amp, s=40)
        #ax.set_xlim(0, T)
        #ax.set_ylim(average(amp) - max(amp)/16, average(amp) + max(amp)/16)
        ax.set_xlabel('Time Step     average amp: '+str(ampAvg))
        ax.set_ylabel('Max Amplitude')
        ax.set_title('Max Amplitude vs. Time Step')

        if inspect:
            plt.show()
        if savePic:
            plt.savefig(str(title))

        return ampAvg

    def simple_plot(dat, id=None):
        '''Simple plot for average max amplitude
        Input:
                dat         (array_like)                data
                id          (str)                       identification number/label
        '''
        fig, ax = plt.subplots()   
        ax.plot(np.arange(len(dat)), dat)
        ax.set_xlabel('Instance')
        ax.set_ylabel('Average Max Amplitude')
        ax.set_title('Average Max Amplitude vs. Instance')
        plt.savefig('Avg_'+str(time.time())+'_'+str(id)+'.png')
        
        return

class operation:
    def num_continue():
        '''Resume a numerical calculation of the Kawahara'''

        return

    def num_batches(n):
        '''Numerical calculation of the Kawahara in n batches
        Testing:
            T_batch = 2
            print('batch progress...')
            for i in tqdm(range(n)):
                T_batch += i 
                t_batch = np.linspace(T_batch - 2, T_batch, 400)
        '''

        return

class analytical:
    '''Analytical stability analysis'''

    def collect_u0(U, param, a1, N=10, damping=False):
        '''Approximate the leading Fourier coefficients of the stationary/stable solution u0
        Input:
                U                   (array)             array of guessed leading terms
                param               (list)              parameters [alpha, beta, sigma]  + gamma if damping is enabled
                a1                  (float)             the initial guess of a1
                N                   (int)               the number of leading terms to consider
                damping             (boolean)           enable damping
        Output:
                leading_terms       (list)              list of leading terms in equation (24)
        Details: 
        u0 stable solution          equation (24) 
        reference                   STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                        OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                            arXiv:1806.08445v1
                                                21 Jun 2018
        code reference              code originally by Nadia Aiaseh        Western University, 2019
                                    lightly modified to fit this program
        '''
        if damping:
            alpha, beta, sigma, gamma = param
        else:
            alpha, beta, sigma = param

        leading_terms = np.zeros(N+2)
        
        v0 = U[0]
        a = U[1::]     
        
        for k in range(N+1):
            sum1=0.
            sum2=0.
            for n in range(k,N+1):
                sum1=sum1+a[n]*a[n-k]                           #summing the a terms in the stationary solution
            for n in range(0,k):
                sum2=sum2+a[n]*a[k-n] 
            if damping:
                leading_terms[k]=((v0*a[k] + 1./2.*sigma*sum1 + 1./2.*sigma*sum2 - alpha*k**2*a[k] + beta*k**4*a[k] - gamma*k**2*a[k]))
            else:
                leading_terms[k]=((v0*a[k] + 1./2.*sigma*sum1 + 1./2.*sigma*sum2 - alpha*k**2*a[k] + beta*k**4*a[k]))
        
        #for the last equation, linearize to optain an equation for speed
        leading_terms[N+1]=-a1+a[1]

        return leading_terms
    
    def optimize_u0Coeff(param, a1, an, N=10, steps=1, spaceResolution=501, L=2*np.pi, damping=False, plot=True, savePic=False):
        '''Optimize the leading Fourier coefficients of the stationary solution u0 - equation (24)
        Input:
                param               (list)              parameters [alpha, beta, sigma]  + gamma if damping is enabled
                a1                  (double)            the initial guess of a1
                an                  (double)            the final guess of a1
                N                   (int)               the number of leading terms to consider
                steps               (int)               the number of steps to take to approach the optimal result
                spaceResolution     (int)               the number of spatial steps
                L                   (float)             the length of the domain
                damping             (boolean)           enable damping
                plot                (boolean)           
                savePic             (boolean)
        Output:
                sol                 (array)             array of optimized leading coefficients in equation (24) with v0 at position 0
                v0                  (float)             wave speed
        Details: 
        u0 stable solution          equation (24) 
        reference                   STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                        OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                            arXiv:1806.08445v1
                                                21 Jun 2018
        code reference              code originally by Nadia Aiaseh        Western University, 2019
                                    lightly modified to fit this program
        '''
        if damping:
            alpha, beta, sigma, gamma = param
        else:
            alpha, beta, sigma = param

        guessed = np.zeros(N+2)
        bifurcationV = np.zeros(steps)
        guessed[0] = alpha - beta
        guessed[1] = a1
        a = np.linspace(a1, an, steps)
        domain = np.linspace(0,L,spaceResolution)
        for k in range(steps):

            leading_terms = fsolve(analytical.collect_u0, guessed, args=(param, a[k], N, damping), xtol=1.e-8)
            sol = leading_terms[1::]
            v0 = leading_terms[0]
            guessed = np.concatenate((v0,leading_terms[1],a[k],leading_terms[3::]),axis=None)           #update guessed coefficients
            
            
            phi = sol[-1]*np.cos(0.*domain)                        #approximated stationary solution value u0 on the periodic domain
            phix = -0.*sol[0]*np.sin(0.*domain)
            ii = 0.
            for aii in sol[1:]:
                ii+=1
                phi+=aii*np.cos(ii*domain)
                phix+=(-ii*aii*np.sin(ii*domain))
                
            bifurcationV[k] = v0
            phi+=-sol[0]
            v0+=(-2.*sol[0])

        if plot or savePic:
            plt.plot(domain, phi)
        if savePic:
            plt.savefig('optimize_u0_coeff_'+str(time.time())+'_a1'+str(max(a[k]))+'.png')
            plt.close()
        if plot:
            plt.show()
            plt.clf()
        
        return sol, v0, phi                  #added u1 solution 

    def fourierCoeffMatrix(param, leadingu0, V, mu, kModes=10, damping=False):
        '''Calculate the Fourier coefficient eigenvector for the perturbation/unstable solution u1
        Input:
                param                       (list)              list of parameters [alpha, beta, sigma]  + gamma if damping is enabled
                leadingu0                   (list)              leading terms of the stable solution u0 - equation (24)
                V                           (float)             wave speed
                mu                          (float)             Floquet parameter mu
                kModes                      (int)               number of leading terms to consider
                damping                     (boolean)           enable damping
        Output:
                lambdaCalc                  (array)             the eigenvalue of matrix S - equation (35)
                U1                          (array)             the eigenvector of matrix S - equation (35)
        Details: 
        u1 stable solution          equation (9) 
        reference                   STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                        OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                            arXiv:1806.08445v1
                                                21 Jun 2018
        code reference              code originally by Nadia Aiaseh        Western University, 2019
                                    lightly modified to fit this program
        '''
        if damping:
            alpha, beta, sigma, gamma = param
        else:
            alpha, beta, sigma = param
       
        #MATRIX OPERATION
        #Matrix D
        matrixD = np.zeros((2*kModes+1, 2*kModes+1), 'complex')
        #Matrix T
        matrixT = np.zeros((2*kModes+1, 2*kModes+1), 'complex')
       
        for m in range(2*kModes+1): 
            for n in range(2*kModes+1):  
                ns = n - kModes   
                ms = m - kModes
                if ns == ms:
                    if damping:
                        matrixD[m, n] = (ns + mu)*V - (ns + mu)**3*alpha + (ns + mu)**5*beta - (ns + mu)**2*gamma   #modified equation (37) with damping
                    else:
                        matrixD[m, n] = (ns + mu)*V - (ns + mu)**3*alpha + (ns + mu)**5*beta            #equivalent to equation (37)   
                else:
                    matrixT[m, n] = 2*sigma*(mu+ms)*leadingu0[abs(ns-ms)]                           #equivalent to equation (38)

        matrixS = 1j*matrixD + 1j*matrixT                                                           #equation (36)
        lambdaCalc, U1 = np.linalg.eig(matrixS)                                                     #equation (35)
        
        return lambdaCalc, U1

    def lambdaFloquet(minMu, maxMu, steps, param, kmode=10, L=2*np.pi, damping=False, plot=True, title=None, savePic=False, inspect=False):
        '''Calculates the Lambda vs. mu in search for the lambda with the largest real part - impies stability
        Input:
                minMu               (float)             the initial mu
                maxMu               (float)             the final mu
                steps               (int)               the number of steps to reach maxMu
                param               (list)              parameters [alpha, beta, sigma] + gamma if damping is enabled
                kMode               (int)               the number of leading terms to consider
                L                   (float)             periodic domain length
                damping             (boolean)           enable damping
                plot                (boolean)           plot
                title               (str)               title
                savePic             (boolean)           save picture
                inspect             (boolean)           display
        Output:
                maxLambda           (array)             the array of maximum lambdas
        Details: 
        u0 stable solution          equation (24) 
        reference                   STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                        OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                            arXiv:1806.08445v1
                                                21 Jun 2018
        code reference              code originally by Nadia Aiaseh        Western University, 2019
                                    lightly modified to fit this program
        Note:
                the guessed amplitude a1 should also be counted in the calculation       
        '''
        mus = np.linspace(minMu, maxMu, steps)     
        maxLambda = np.zeros((steps,1))
        a1 =10**(-2)     #a1 = epsilon
        an = 10**(-2)    #aq == an; step = 1 for optimize

        leading_terms_u0, v0, _ = analytical.optimize_u0Coeff(param, a1, an, steps=1, plot=False, N=2*kmode+1, L=L, damping=damping)
        for i in range(steps):
            lambdaCalc, U1 = analytical.fourierCoeffMatrix(param, leading_terms_u0, v0, mus[i], damping=damping)
            maxLambda[i] = np.max(lambdaCalc.real)              #then use equation 9 to get u1

        if plot or savePic:
            fig, ax = plt.subplots()   #initialize figure
            ax.scatter(mus, maxLambda)
            ax.set_xlabel('Floquet Parameter mu')
            ax.set_ylabel('Max Re Lambda Value')
            ax.set_title('Max Lambda Value vs. Floquet Parameter')
            if savePic:
                plt.savefig('lambdaFloquet_'+str(time.time())+'mu_'+ str(mus[int(steps/2)])+'_lambda'+str(max(maxLambda))+'damping'+str(damping)+'.png')
            if inspect:
                plt.show()

        return maxLambda
    
    def find_stable_lamb(param, mu, steps, savePic=False, plot=False):
        '''Find the nearest stable Re{lambda} corresponding to the input mu value
        Imput:
                param               (list)                parameters [alpha, beta, sigma] + gamma if damping is enabled
                mu                  (float)               the Floquet parameter
                steps               (int)                 the number of Floquet parameters within the range to be tested
                savePic             (boolean)
                plot                (boolean)
        Output:
                lambda_damped       (float)               the nearest stable Re{lambda} corresponding to the input mu value
        Note:
                the guessed amplitude a1 should also be counted in the calculation       
        '''
        lamb_damped = analytical.lambdaFloquet(0.97*mu, 1.03*mu, steps, 
                                        param, savePic=False, plot=False, damping=True)
        maxSearch = 0
        while lamb_damped[int(steps/2)] > 10**(-8):
            param[3] = param[3]*2
            lamb_damped = analytical.lambdaFloquet(0.97*mu, 1.03*mu, steps, 
                                    param, savePic=False, plot=False, damping=True)
            maxSearch+=1
            if maxSearch == 50:
                print('Unable to find the damped lambda in 50 searches...')
                break
        print('The damping parameter gamma is: '+str(param[3]))
        lamb_damped = analytical.lambdaFloquet(0.97*mu, 1.03*mu, steps, 
                                    param, savePic=savePic, plot=plot, damping=True)

        return lamb_damped[int(steps/2)]
    
    def collect_u1(lambdaCalc, U1, mu, x):
        '''Calculate the perturbation solution u1 - equation (9)
        Input:
                lambdaCalc              (array)             the array of lambdas
                U1                      (array)             the array of eigenvectors
                mu                      (float)             the value of the Floquet parameter mu
                x                       (array)             the spatial domain
        Output:
                combined                (float)             the perturbation solution u1
        Details: 
        u1 perturbation solution        equation (9) 
        reference                       STABILITY OF PERIODIC TRAVELLING WAVE SOLUTIONS TO THE KAWAHARA EQUATION
                                            OLGA TRICHTCHENKO, BERNARD DECONINCK, AND RICHARD KOLLAR
                                                arXiv:1806.08445v1
                                                    21 Jun 2018
        '''
        #find the max lambda position
        lambdaCalcMax = max(np.real(lambdaCalc))
        indVec = np.argwhere(np.real(lambdaCalc)==lambdaCalcMax)
        U1 = np.real(U1[:,indVec].transpose())
        sumPostive = 0
        sumNegative = 0
        for i in range(len(lambdaCalc)):
            sumPostive += U1[0][0][i]*np.exp(1j*(mu+i)*x)                                #equivalent to equation (9)
            sumNegative += U1[0][0][i]*np.exp(1j*(mu-i)*x)                               #equivalent to equation (9)

        #equation (9) break down
        #u1 = b1*np.exp(1j*(mu-1)*x) + b2*np.exp(1j*(mu-2)*x) + b3*np.exp(1j*(mu-3)*x) + b4*np.exp(1j*(mu-4)*x) + \
                #b1*np.exp(1j*(mu+1)*x) + b2*np.exp(1j*(mu+2)*x) + b3*np.exp(1j*(mu+3)*x) + b4*np.exp(1j*(mu+4)*x) 
        combined = sumPostive+sumNegative

        return combined

if __name__ == "__main__":
    test()
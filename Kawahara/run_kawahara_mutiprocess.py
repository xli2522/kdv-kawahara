#from fourierKawahara_num_analysis import *
from hardware_accelerated.fourierKawahara_num_analysis_mutiprocess_accelerated import *
#############     Numerical Instability   #############
#############                             #############

#(1) Group 1
#mus = [0.69]
#lambs = [-57.60]

def full_operation(pair):
    mu = pair[0]
    lamb = pair[1]
    beta = 3/160
    a1= 0.000001
    #a1=0.001
    param1 = [1, beta, 1, 0.01, lamb]                               #[alpha, beta, sigma, epsilon, lamb]
    param2 = [0.001, lamb, mu, 1, beta, 1]                           #[delta, lamb, mu, alpha, beta, sigma]
    #a1 = 0.373
    damp_all_cases = False
    damp_all_cases_analytical = False
    optimize_stable = False
    optimize_unstable = False

    main_start = time.time()

    #####################################################################
    # Set the size of the domain, and create the discretized grid.
    L = 10*np.pi
    #force a larger periodic domain
    '''
    if L >= 10*np.pi:
        L = L
    else:
        L = 10*np.pi'''
    
    N = int(np.floor(30*L/(2*np.pi)))       #number of spatial steps; fit to the length of the periodic domain
    dx = L / (N - 1.0)                      #spatial step size
    x = np.linspace(0, (1-1.0/N)*L, N)      #initialize x spatial axis    
    # Set the time sample grid.
    T = 0.01
    t = np.linspace(0, T, 1000)
    dt = len(t)
    
    ######################################################################
    param_damping = [1, beta, 1, a1]
    param_no_damping = [1, beta, 1]
       
    if damp_all_cases_analytical:
        #this is incorrect atm; find_stable_lamb outputs the Re{lambda} instead of the Im{lambda}
        lamb, gamma = analytical.find_stable_lamb(param_damping, a1, mu, 800, savePic=True, plot=False)
        param_damping = [1, beta, 1, gamma]
        '''if lambs[i] > lamb:
            lamb = lambs[i]
        else:
            lamb = lamb'''
    
        mu = mu

    else:
        lamb = lamb
        mu = mu
    #continue
    
    if optimize_unstable:
        a1 = analytical.optimize_Floquet_a1(param_no_damping, 0.1, 1.1, mu, 100, stable=False, savePic=True)
    elif optimize_stable:
        a1 = analytical.optimize_Floquet_a1(param_no_damping, 0.0009, 0.1, mu, 1000, stable=True, savePic=True)
    else:
        a1 = a1
        lamb_no_dampRe, lamb_no_dampIm = analytical.lambdaFloquet(0.97*mu, 1.03*mu, 200, a1, param_no_damping, mu, savePic=False, damping=damp_all_cases_analytical)

    if a1 == None:
        pass
    #calculate Re{lambda} values in the Floquet-Re{lambda} space
    #lamb_original = analytical.lambdaFloquet(0.97*mu, 1.03*mu, 200, a1,
                                    #param_no_damping, mu, savePic=True, plot=False, damping=False)     
    #continue
    #####################################################################
    #a1 = 0.001   #param1[3]                                        #DECREASE A1 WHEN BETA IS LARGE

    kModes=10                                                       #number of fourier modes to consider
    ic_start = time.time()
    
    ###########################     U0  aN    ###########################
    #approximate the optimized u0 coeficients for analytical approximation of perturbation solution u1
    if damp_all_cases_analytical:
        optimized_u0, v0, stationary_u0 = analytical.optimize_u0Coeff(param_damping, a1, a1, 
                        steps=1, L=L, spaceResolution=N, plot=False, N=2*kModes+1, savePic=False, damping=damp_all_cases_analytical)
        lambdaCalc, U1 = analytical.fourierCoeffMatrix(param_damping, optimized_u0, v0, mu, damping=damp_all_cases_analytical)
    else:
        optimized_u0, v0, stationary_u0 = analytical.optimize_u0Coeff(param_no_damping, a1, a1, 
                        steps=1, L=L, spaceResolution=N, plot=False, N=2*kModes+1, savePic=False, damping=damp_all_cases_analytical)

        ###########################     U1  aN    ###########################
        #calculate the leading terms of the perturbation solution u1
        lambdaCalc, U1 = analytical.fourierCoeffMatrix(param_no_damping, optimized_u0, v0, mu, kModes=kModes, damping=damp_all_cases_analytical)
    
    #calculate the perturbation solution u1
    perturbation_u1 = analytical.collect_u1(lambdaCalc, lamb, U1, mu, x)
    
    ###########################     Uc  wE    ###########################
    #calculate the combined solution u
    combined_u =  waveEquation.kawahara_combined_solution(stationary_u0, x, param2, a1, a1, u1=perturbation_u1)       

    #plot the combined initial condition    
    visual.plotInitial(combined_u, mu, a1)              

    #calculate the FFT of the initial conditions
    t = np.arange(N)
    sp = np.fft.fft(combined_u)
    freq = np.fft.fftfreq(t.shape[-1])
    plt.plot(freq, sp.real)#, freq, sp.imag)
    plt.savefig('fft_'+str(int(L/np.pi))+'pi_'+str(mu)+'mu'+'.png')
    plt.clf()
    
    ###########################     Solve     ###########################
    print("Initial condition calculation --- %s seconds ---" % (time.time() - ic_start))
    solver_start = time.time()
    sol = waveEquation.solve_kawahara(waveEquation.kawahara_model, combined_u, t, L, param1, 3000, modelArg=(True, damp_all_cases, 0.1, v0))
    print("Numerical solver --- %s seconds ---" % (time.time() - solver_start))
    print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))

    avg = numAnalysis.amplitude(sol, len(t), title='Table_test_'+str(a1)+'maxamp_'+str(mu)+'_'+str(int(L/np.pi))+'pi'+'.png', ampType='central')
    #continue
    #visual.plot_video(sol, len(t), N, L, 'Table_test_'+str(int(L/np.pi))+'pi_'+str(mu)+'mu'+ '.avi', fps=30)
    
    #############                             #############
    #############     Numerical Instability   #############
    return 

if __name__ == '__main__':
    #mus = [5.09,5.48,4.79,5.02,4.16,4.23,3.24,-3.23,2.27,2.36,1.49,1.64,0.74,0.69]
    #lambs = [62.82,51.62,35.98,19.78,7.09,0.595,-0.219,-0.305,-3.22,-13.41,-29.71,-49.13,-64.87,-57.60]
    current = time.time()
    pairs1 = [[0.7845, -0.1798], [0.6324,0.2277 ], [-0.7928, 0.2128]]       #beta = 1/4
    pairs2 = [[5.09,62.82], [5.48,51.62], [4.79,35.98], [5.02,19.78], [4.16,7.09], [4.23,0.595], [3.24,-0.219], \
                    [-3.23,-0.305], [2.27,-3.22], [2.36,-13.41], [1.49,-29.71], [1.64,-49.13], [0.74,-64.87], [0.69, -57.60]]           #beta = 3/160
    
    pool = Pool(processes=14)
    pool.map(full_operation, pairs2)
    pool.terminate()
    print('Full operation completed in ---- ' + str(time.time() - current) + ' ----')

    #analytical.numError(5.09, 62.82, 3/160, 0.00001, 800, 3, 8)         # n1/n0 = int; n1 = intiger multiples of n0
#(2) Group 2
#beta = 1/4    
#mus = [0.7845, 0.6324, -0.7928]
#lambs = [-0.1798, 0.2277, 0.2128] 

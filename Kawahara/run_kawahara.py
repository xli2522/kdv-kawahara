from fourierKawahara_num_analysis import *

#############     Numerical Instability   #############
#############                             #############

#(1) Group 1
beta = 3/160
mus = [5.09,5.48,4.79,5.02,4.16,4.23,3.24,-3.23,2.27,2.36,1.49,1.64,0.74,0.69]
lambs = [62.82,51.62,35.98,19.78,7.09,0.595,-0.219,-0.305,-3.22,-13.41,-29.71,-49.13,-64.87,-57.60]

#(2) Group 2
#beta = 1/4    
#mus = [0.7845, 0.6324, -0.7928]
#lambs = [-0.1798, 0.2277, 0.2128] 

a1=0.001
damp_all_cases = False
optimize_stable = True
optimize_unstable = False

avg = np.zeros(len(mus))       
main_start = time.time()

for i in tqdm(range(len(mus))):

    #####################################################################
    # Set the size of the domain, and create the discretized grid.
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
    T = 4
    t = np.linspace(0, T, 3200)
    dt = len(t)

    ######################################################################
    param_damping = [1, beta, 1, 0.01]
    param_no_damping = [1, beta, 1]
       
    if damp_all_cases:
        #this is incorrect atm; find_stable_lamb outputs the Re{lambda} instead of the Im{lambda}
        lamb, gamma = analytical.find_stable_lamb(param_damping, a1, mus[i], 800, savePic=True, plot=False)
        param_damping = [1, beta, 1, gamma]
        '''if lambs[i] > lamb:
            lamb = lambs[i]
        else:
            lamb = lamb'''
    
        mu = mus[i]

    else:
        lamb = lambs[i]
        mu = mus[i]
    #continue

    if optimize_stable or optimize_unstable:
        if optimize_unstable:
            a1 = analytical.optimize_Floquet_a1(param_no_damping, 0.001, 1, mus[i], 1000, stable=False, savePic=True)
        else:
            a1 = analytical.optimize_Floquet_a1(param_no_damping, 0.001, 1, mus[i], 1000, stable=True, savePic=True)
    
    #calculate Re{lambda} values in the Floquet-Re{lambda} space
    #lamb_original = analytical.lambdaFloquet(0.97*mu, 1.03*mu, 200, a1,
                                    #param_no_damping, mu, savePic=True, plot=False, damping=False)     
    #continue
    ######################################################################
    param1 = [1, beta, 1, 0.01, lamb]                               #[alpha, beta, sigma, epsilon, lamb]
    param2 = [1, lamb, mu, 1, beta, 1]                           #[delta, lamb, mu, alpha, beta, sigma]
    #a1 = 0.001   #param1[3]                                        #DECREASE A1 WHEN BETA IS LARGE

    kModes=10                                                       #number of fourier modes to consider
    ic_start = time.time()
    
    ###########################     U0  aN    ###########################
    #approximate the optimized u0 coeficients for analytical approximation of perturbation solution u1
    if damp_all_cases:
        optimized_u0, v0, stationary_u0 = analytical.optimize_u0Coeff(param_damping, a1, a1, 
                        steps=1, L=L, spaceResolution=N, plot=False, N=2*kModes+1, damping=damp_all_cases)
        lambdaCalc, U1 = analytical.fourierCoeffMatrix(param_damping, optimized_u0, v0, mu, damping=damp_all_cases)
    else:
        optimized_u0, v0, stationary_u0 = analytical.optimize_u0Coeff(param_no_damping, a1, a1, 
                        steps=1, L=L, spaceResolution=N, plot=False, N=2*kModes+1, damping=damp_all_cases)

        ###########################     U1  aN    ###########################
        #calculate the leading terms of the perturbation solution u1
        lambdaCalc, U1 = analytical.fourierCoeffMatrix(param_no_damping, optimized_u0, v0, mu, kModes=kModes, damping=damp_all_cases)
    
    #calculate the perturbation solution u1
    perturbation_u1 = analytical.collect_u1(lambdaCalc, lamb, U1, mu, x)
    
    ###########################     Uc  wE    ###########################
    #calculate the combined solution u
    
    combined_u =  waveEquation.kawahara_combined_solution(stationary_u0, x, param2, 0.01, a1, u1=perturbation_u1)           
    # , leading_terms_u1=waveEquation.u1_leading_terms()
    #visual.plotInitial(combined_u, mu, a1, i)
    
    #fig, axs = plt.subplots(4)
    #axs[0].plot(stationary_u0)
    #axs[1].plot(perturbation_u1)
    #axs[2].plot(combined_u)
    #axs[3].plot(stationary_u0+perturbation_u1)
    #plt.show()

    #continue
    print('\n')             #prevent tqdm from consuming the first printed character in terminal                                     
    ###########################     Solve     ###########################
    print("Initial condition calculation --- %s seconds ---" % (time.time() - ic_start))
    solver_start = time.time()
    sol = waveEquation.solve_kawahara(waveEquation.kawahara_model, combined_u, t, L, param1, 3000, modelArg=(True, damp_all_cases, v0))
    print("Numerical solver --- %s seconds ---" % (time.time() - solver_start))
    print("Main numerical simulation --- %s seconds ---" % (time.time() - main_start))
    
    # SAVE THE FULL SOLUTION
    with open('Table_test_'+str(int(L/np.pi))+'pi_'+str(mus[i])+'mu_'+str(i)+'_'+str(T)+'time'+str(len(t))+'steps'+'.txt', "w") as f:
        for row in sol:
            f.write(str(row))

    avg[i] = numAnalysis.amplitude(sol, len(t), title='Table_test_'+str(a1)+'maxamp_'+str(mus[i])+'_'+str(int(L/np.pi))+'pi_'+str(i)+'.png')
    #continue
    visual.plot_video(sol, len(t), N, L, 'Table_test_'+str(int(L/np.pi))+'pi_'+str(mus[i])+'mu_'+str(i)+ '.avi', fps=30)
    
    #############                             #############
    #############     Numerical Instability   #############
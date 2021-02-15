from landscape_evolution import *            #Import the functions



def main():

    N = 5                                    #Landscape size is 2^N
    sigma = 1.0                              #Epistasis
    correl = np.linspace(-1.0,1.0,51)        #Define what correlations you'd like to make in the B landscape.
    saveBs = np.zeros((51,2**N))

    phenom = 0

    A = Landscape(N, sigma)                                                      #Generate A Landscapes.
    Bs = A.generate_correlated_landscapes(correl, without_shared_max=False)      #Generate B landscapes. Note the flag here. This doesn't tamper, however if you set to True you'll only generate landscapes with no shared maxima.

    for k in range(len(correl)):
        saveBs[k,:] = Bs[k].ls

    saveA = np.asarray(A.ls)
    print(saveBs)

    np.savetxt("BsForSteph.csv", saveBs, delimiter=",")
    np.savetxt("AForSteph.csv",saveA,delimiter=",")


if __name__ == '__main__':
    main()

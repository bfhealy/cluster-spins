import numpy as np
import statsmodels.distributions.empirical_distribution as emp
import time
import matplotlib.pyplot as plt
from scipy.stats import t

def true_sini_distrib_specific(alpha,lamda,n=50000):
    alpha *= np.pi/180
    lamda *= np.pi/180

    Rn = np.random.random(n)
    Rnp = np.random.random(n)

    theta_n = np.arccos(1 - Rn*(1-np.cos(lamda)))
    Ptheta = (1 - np.cos(theta_n))/(1 - np.cos(lamda))
    phi_n = 2*np.pi*Rnp

    cosi_n = np.sin(alpha)*np.sin(theta_n)*np.cos(phi_n) + np.cos(alpha)*np.cos(theta_n)
    sini_n = np.sin(np.arccos(np.abs(cosi_n)))

    sini_n = np.sort(sini_n)

    return sini_n

def sini_distrib_interp(xdatapoints,alpha,lamda,n=50000):
    sini_n_true = true_sini_distrib_specific(alpha,lamda,n=n)

    if len(sini_n_true) != 0:
        mcincemp = emp.ECDF(sini_n_obs)
        mcincempx1 = mcincemp.x[1:]
        mcincempy1 = mcincemp.y[1:]
    else:
        mcincempy1 = np.zeros(len(xdatapoints)) + 1
        mcincempx1 = xdatapoints
        #mcincemp = emp.ECDF(sini_n_true)
    sini_n_obs = np.interp(xdatapoints,mcincempx1,mcincempy1)



    return sini_n_obs

def true_sini_distrib(alpha,lamda,n=50000):
    alpha *= np.pi/180
    lamda *= np.pi/180

    Rn = np.random.random(n)
    Rnp = np.random.random(n)

    theta_n = np.arccos(1 - Rn*(1-np.cos(lamda)))
    Ptheta = (1 - np.cos(theta_n))/(1 - np.cos(lamda))
    phi_n = 2*np.pi*Rnp

    cosi_n = np.sin(alpha)*np.sin(theta_n)*np.cos(phi_n) + np.cos(alpha)*np.cos(theta_n)
    sini_n = np.sin(np.arccos(np.abs(cosi_n)))

    return sini_n



def obs_sini_distrib(alpha,lamda,deltap=0.1,deltav=0.1,deltar=0.1,cutoff=0,n=50000):
    sini_n_true = true_sini_distrib(alpha,lamda,n=n)

    U1 = np.random.normal(size=n)
    U2 = np.random.normal(size=n)

    deltapv = np.sqrt(deltap**2+deltav**2)

    sini_n_obs = sini_n_true #* ((1 + deltapv*U1)/(1 + deltar*U2))

    cutoffmask = np.zeros(len(sini_n_obs),dtype=bool)
    for i in range(len(sini_n_obs)):
        if sini_n_obs[i] > cutoff:
            cutoffmask[i] = True

    sini_n_obs = sini_n_obs[cutoffmask]

    return sini_n_obs

def true_sini_distrib_specific(alpha,lamda,n):
    alpha *= np.pi/180
    lamda *= np.pi/180

    Rn = np.random.random(n)
    Rnp = np.random.random(n)

    theta_n = np.arccos(1 - Rn*(1-np.cos(lamda)))
    Ptheta = (1 - np.cos(theta_n))/(1 - np.cos(lamda))
    phi_n = 2*np.pi*Rnp

    cosi_n = np.sin(alpha)*np.sin(theta_n)*np.cos(phi_n) + np.cos(alpha)*np.cos(theta_n)
    sini_n = np.sin(np.arccos(np.abs(cosi_n)))

    sini_n = np.sort(sini_n)

    return sini_n

def obs_sini_distrib_specific(xdatapoints,alpha,lamda,deltasini=0.11641160175478454,deltap=0.038906455369088946,deltav=0.05176470588235295,deltar=0.023155991748531755,cutoff=0.,n=50000):
    sini_n_true = true_sini_distrib_specific(alpha,lamda,n)

    U1 = np.random.normal(size=n)
    U2 = np.random.normal(size=n)

    deltapv = np.sqrt(deltap**2+deltav**2)

    sini_n_obs = sini_n_true #* ((1 + deltapv*U1)/(1 + deltar*U2))
    #print(sini_n_obs)

    cutoffmask = np.zeros(len(sini_n_obs),dtype=bool)
    for i in range(len(sini_n_obs)):
        if sini_n_obs[i] > cutoff:
            cutoffmask[i] = True

    sini_n_obs = sini_n_obs[cutoffmask]
    if len(sini_n_obs) != 0:
        mcincemp = emp.ECDF(sini_n_obs)
        mcincempx1 = mcincemp.x[1:]
        mcincempy1 = mcincemp.y[1:]
    else:
        mcincempy1 = np.zeros(len(xdatapoints)) + 1
        mcincempx1 = xdatapoints


    #sini_n_obs = np.interp(xdatapoints,mcincemp.x[1:],mcincemp.y[1:])
    sini_n_obs = np.interp(xdatapoints,mcincempx1,mcincempy1)


    return sini_n_obs

def obs_sini_distrib_uncerts(xdatapoints,alpha,lamda,deltasini=0.110/2,deltap=0.038906455369088946,deltav=0.05176470588235295,deltar=0.023155991748531755,cutoff=0.,n=50000,lognorm=False):
    xdatapoints = np.sort(xdatapoints)
    sini_n_true = true_sini_distrib_specific(alpha,lamda,n)
    sini_n_true = sini_n_true[sini_n_true > cutoff]

    n = len(sini_n_true)

    U1 = np.random.normal(size=n)
    U2 = np.random.normal(size=n)

    deltapv = np.sqrt(deltap**2+deltav**2)

    #print(np.mean((1 + deltapv*U1)/(1 + deltar*U2)))
    if lognorm == False:
        sini_n_obs = sini_n_true * ((1 + deltapv*U1)/(1 + deltar*U2))
    elif lognorm == True:
        U3 = np.random.lognormal(size=n)
        plusorminus = np.random.choice([-1,1],n)
        #U4 = np.random.normal(size=n)
        sini_n_obs = sini_n_true * ((1 + deltap*U1) * (1 + deltav*U3*plusorminus) / (1 + deltar*U2))

        #each star has fractional vsini uncertainty such that the distribution is lognormal
        #one star: draw from lognormal distribution to get fractional uncertainty
        #this number is the +/- fractional uncertainty -- how to make it negative?

    #sini_n_obs = sini_n_true * (1 + deltasini * U1)

    #print(xdatapoints)
    #print(sini_n_obs)

    #cutoffmask = np.zeros(len(sini_n_obs),dtype=bool)
    upperlimit = 3

    keepthese = (sini_n_obs < upperlimit) & (sini_n_obs > 0)
    sini_n_obs = sini_n_obs[keepthese]

    #for i in range(len(sini_n_obs)):
        ###if sini_n_obs[i] > cutoff:
    #    if (sini_n_true[i] > cutoff) & (sini_n_obs[i] < upperlimit) & (sini_n_obs[i] > 0):
    #        cutoffmask[i] = True

    #sini_n_obs = sini_n_obs[cutoffmask]
    if len(sini_n_obs) != 0:
        mcincemp = emp.ECDF(sini_n_obs)
        mcincempx1 = mcincemp.x[1:]
        mcincempy1 = mcincemp.y[1:]
    else:
        mcincempy1 = np.zeros(len(xdatapoints)) + 1
        mcincempx1 = xdatapoints

    #print(mcincempx1)
    #print(mcincempy1)
    #sini_n_obs = np.interp(xdatapoints,mcincemp.x[1:],mcincemp.y[1:])
    #sini_n_obs = np.interp(xdatapoints,mcincempx1,mcincempy1)


    return sini_n_obs

def obs_sini_distrib_uncerts_updated(xdatapoints,alpha,lamda,deltasini=0.110/2,deltap=0.038906455369088946,d_lnV=0.05176470588235295,sig_lnV=0.5,deltar=0.023155991748531755,deltav=0.1,cutoff=0.,n=50000,lognorm=False):
        xdatapoints = np.sort(xdatapoints)
        sini_n_true = true_sini_distrib_specific(alpha,lamda,n)
        sini_n_true = sini_n_true[sini_n_true > cutoff]

        n = len(sini_n_true)

        U1 = np.random.normal(size=n)
        U2 = np.random.normal(size=n)

        #print(np.mean((1 + deltapv*U1)/(1 + deltar*U2)))
        if lognorm == False:
            deltapv = np.sqrt(deltap**2+deltav**2)
            sini_n_obs = sini_n_true * ((1 + deltapv*U1)/(1 + deltar*U2))
        elif lognorm == True:
            U3 = np.random.lognormal(d_lnV, sig_lnV, size=n)
            plusorminus = np.random.choice([-1,1],n)

                #sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + plusorminus*sim_e_vsini_frac) / (1 + deltaR*U2))
            sini_n_obs = sini_n_true * ((1 + deltap*U1) * (1 + plusorminus*U3) / (1 + deltar*U2))

            nonzero_sini_obs = sini_n_obs > 0
            sini_n_obs = sini_n_obs[nonzero_sini_obs]
            #each star has fractional vsini uncertainty such that the distribution is lognormal
            #one star: draw from lognormal distribution to get fractional uncertainty
            #this number is the +/- fractional uncertainty -- how to make it negative?

        #sini_n_obs = sini_n_true * (1 + deltasini * U1)

        #print(xdatapoints)
        #print(sini_n_obs)

        #cutoffmask = np.zeros(len(sini_n_obs),dtype=bool)
        #upperlimit = 3

        #keepthese = (sini_n_obs < upperlimit) & (sini_n_obs > 0)
        #sini_n_obs = sini_n_obs[keepthese]

        #for i in range(len(sini_n_obs)):
            ###if sini_n_obs[i] > cutoff:
        #    if (sini_n_true[i] > cutoff) & (sini_n_obs[i] < upperlimit) & (sini_n_obs[i] > 0):
        #        cutoffmask[i] = True

        #sini_n_obs = sini_n_obs[cutoffmask]
        if len(sini_n_obs) != 0:
            mcincemp = emp.ECDF(sini_n_obs)
            mcincempx1 = mcincemp.x[1:]
            mcincempy1 = mcincemp.y[1:]
        else:
            mcincempy1 = np.zeros(len(xdatapoints)) + 1
            mcincempx1 = xdatapoints

        #print(mcincempx1)
        #print(mcincempy1)
        #sini_n_obs = np.interp(xdatapoints,mcincemp.x[1:],mcincemp.y[1:])
        #sini_n_obs = np.interp(xdatapoints,mcincempx1,mcincempy1)


        return sini_n_obs

def obs_sini_distrib_uncerts_kde(xdatapoints,alpha,lamda,gkde,deltasini=0.110/2,deltap=0.038906455369088946,deltav=0.05176470588235295,deltar=0.023155991748531755,cutoff=0.,n=50000):
    xdatapoints = np.sort(xdatapoints)
    sini_n_true = true_sini_distrib_specific(alpha,lamda,n)

    #U1 = np.random.normal(size=n)
    U1 = gkde.resample(size=n)[0]
    U2 = np.random.normal(size=n)

    deltapv = np.sqrt(deltap**2+deltav**2)

    #print(np.mean((1 + deltapv*U1)/(1 + deltar*U2)))
    sini_n_obs = sini_n_true * ((1 + U1)/(1 + deltar*U2))

    #sini_n_obs = sini_n_true * ((1 + deltapv*U1)/(1 + deltar*U2))
    #sini_n_obs = sini_n_true * (1 + deltasini * U1)

    #print(xdatapoints)
    #print(sini_n_obs)

    cutoffmask = np.zeros(len(sini_n_obs),dtype=bool)
    for i in range(len(sini_n_obs)):
        if sini_n_obs[i] > cutoff:
            cutoffmask[i] = True

    sini_n_obs = sini_n_obs[cutoffmask]
    if len(sini_n_obs) != 0:
        mcincemp = emp.ECDF(sini_n_obs)
        mcincempx1 = mcincemp.x[1:]
        mcincempy1 = mcincemp.y[1:]
    else:
        mcincempy1 = np.zeros(len(xdatapoints)) + 1
        mcincempx1 = xdatapoints

    #print(mcincempx1)
    #print(mcincempy1)
    #sini_n_obs = np.interp(xdatapoints,mcincemp.x[1:],mcincemp.y[1:])
    #sini_n_obs = np.interp(xdatapoints,mcincempx1,mcincempy1)


    return sini_n_obs

def obs_sini_distrib_uncerts_specific(xdatapoints,alpha,lamda,deltasini=0.11641160175478454,deltap=0.038906455369088946,deltav=0.05176470588235295,deltar=0.023155991748531755
,cutoff=0.,n=50000):
    xdatapoints = np.sort(xdatapoints)
    sini_n_true = true_sini_distrib_specific(alpha,lamda,n)

    U1 = np.random.normal(size=n)
    U2 = np.random.normal(size=n)

    deltapv = np.sqrt(deltap**2+deltav**2)
    #print((1 + deltapv*U1)/(1 + deltar*U2))

    sini_n_obs = sini_n_true * ((1 + deltapv*U1)/(1 + deltar*U2))
    #print(xdatapoints)
    #print(sini_n_obs)

    cutoffmask = np.zeros(len(sini_n_obs),dtype=bool)
    upperlimit=3
    for i in range(len(sini_n_obs)):
        #if sini_n_obs[i] > cutoff:
        if (sini_n_true[i] > cutoff) & (sini_n_obs[i] < upperlimit) & (sini_n_obs[i] > 0):
            cutoffmask[i] = True

    sini_n_obs = sini_n_obs[cutoffmask]
    if len(sini_n_obs) != 0:
        mcincemp = emp.ECDF(sini_n_obs)
        mcincempx1 = mcincemp.x[1:]
        mcincempy1 = mcincemp.y[1:]
    else:
        mcincempy1 = np.zeros(len(xdatapoints)) + 1
        mcincempx1 = xdatapoints

    #print(mcincempx1)
    #print(mcincempy1)
    #sini_n_obs = np.interp(xdatapoints,mcincemp.x[1:],mcincemp.y[1:])
    sini_n_obs = np.interp(xdatapoints,mcincempx1,mcincempy1)


    return sini_n_obs

def mc_sini_distrib(sim_vrot, alpha, lamda, d_lnV, sig_lnV, deltaP, deltaR, vsini_threshold=0, interp=False, xdatapoints=None):
    t0 = time.time()

    rand_inclination_distrib = true_sini_distrib(alpha, lamda, n=len(sim_vrot))

    sim_vsini = sim_vrot * rand_inclination_distrib

    U1 = np.random.normal(size=len(rand_inclination_distrib))
    U2 = np.random.normal(size=len(rand_inclination_distrib))

    #sim_e_vsini = np.random.lognormal(d_lnV, sig_lnV, size=len(rand_inclination_distrib))
    #sim_e_vsini_frac = sim_e_vsini / sim_vsini

    U3 = np.random.lognormal(d_lnV, sig_lnV, size=len(rand_inclination_distrib))
    plusorminus = np.random.choice([-1,1],len(rand_inclination_distrib))

    #sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + plusorminus*sim_e_vsini_frac) / (1 + deltaR*U2))
    sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + plusorminus*U3) / (1 + deltaR*U2))

    nonzero_sini_obs = sini_n_obs > 0
    sini_n_obs = sini_n_obs[nonzero_sini_obs]
    #print(np.min(sini_n_obs),np.max(sini_n_obs))

    scattered_incs_good_vsini = sini_n_obs[sim_vsini[nonzero_sini_obs] > vsini_threshold]

    if interp==True:
        if len(scattered_incs_good_vsini) != 0:
            mcincemp = emp.ECDF(scattered_incs_good_vsini)
            mcincempx1 = mcincemp.x[1:]
            mcincempy1 = mcincemp.y[1:]
        else:
            mcincempy1 = np.zeros(len(xdatapoints)) + 1
            mcincempx1 = xdatapoints

        final_cdf_vals = np.interp(xdatapoints,mcincempx1,mcincempy1)
    else:
        final_cdf_vals = scattered_incs_good_vsini

    return final_cdf_vals

def mc_sini_distrib_test(sim_vrot, alpha, lamda, d_lnV, sig_lnV, deltaP, deltaR, vsini_threshold=0, interp=False, xdatapoints=None):
        t0 = time.time()

        rand_inclination_distrib = true_sini_distrib(alpha, lamda, n=len(sim_vrot))

        sim_vsini = sim_vrot * rand_inclination_distrib

        U1 = np.random.normal(size=len(rand_inclination_distrib))
        U2 = np.random.normal(size=len(rand_inclination_distrib))

        #sim_e_vsini = np.random.lognormal(d_lnV, sig_lnV, size=len(rand_inclination_distrib))
        #sim_e_vsini_frac = sim_e_vsini / sim_vsini

        U3 = np.random.lognormal(d_lnV, sig_lnV, size=len(rand_inclination_distrib))
        plusorminus = np.random.choice([-1,1],len(rand_inclination_distrib))

        #sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + plusorminus*sim_e_vsini_frac) / (1 + deltaR*U2))

        sini_n_obs_before = rand_inclination_distrib * ((1 + deltaP*U1) / (1 + deltaR*U2))

        U4 = np.random.normal(np.zeros(len(rand_inclination_distrib)), U3)

        sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + U4) / (1 + deltaR*U2))

        #sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + plusorminus*U3) / (1 + deltaR*U2))


        plt.scatter(sini_n_obs_before, sini_n_obs)
        plt.show()

        nonzero_sini_obs = sini_n_obs > 0
        sini_n_obs = sini_n_obs[nonzero_sini_obs]
        #print(np.min(sini_n_obs),np.max(sini_n_obs))

        scattered_incs_good_vsini = sini_n_obs[sim_vsini[nonzero_sini_obs] > vsini_threshold]

        if interp==True:
            if len(scattered_incs_good_vsini) != 0:
                mcincemp = emp.ECDF(scattered_incs_good_vsini)
                mcincempx1 = mcincemp.x[1:]
                mcincempy1 = mcincemp.y[1:]
            else:
                mcincempy1 = np.zeros(len(xdatapoints)) + 1
                mcincempx1 = xdatapoints

            final_cdf_vals = np.interp(xdatapoints,mcincempx1,mcincempy1)
        else:
            final_cdf_vals = scattered_incs_good_vsini

        return final_cdf_vals

def mc_sini_distrib_new(sim_vrot, alpha, lamda, d_lnV, sig_lnV, deltaP, deltaR, vsini_threshold=0, interp=False, xdatapoints=None, ydatapoints=None,tdistrib=False):
    t0 = time.time()

    rand_inclination_distrib = true_sini_distrib(alpha, lamda, n=len(sim_vrot))

    sim_vsini = sim_vrot * rand_inclination_distrib

    U1 = np.random.normal(size=len(rand_inclination_distrib))
    U2 = np.random.normal(size=len(rand_inclination_distrib))

    #sim_e_vsini = np.random.lognormal(d_lnV, sig_lnV, size=len(rand_inclination_distrib))
    #sim_e_vsini_frac = sim_e_vsini / sim_vsini

    U3 = np.random.lognormal(d_lnV, sig_lnV, size=len(rand_inclination_distrib))
    #plusorminus = np.random.choice([-1,1],len(rand_inclination_distrib))

    if not tdistrib:
        U4 = np.random.normal(np.zeros(len(rand_inclination_distrib)), U3)
    else:
        U4 = t.rvs(loc=np.zeros(len(rand_inclination_distrib)), scale=U3, df=2)

    sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + U4) / (1 + deltaR*U2))

    #sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + plusorminus*sim_e_vsini_frac) / (1 + deltaR*U2))
    #sini_n_obs = rand_inclination_distrib * ((1 + deltaP*U1) * (1 + plusorminus*U3) / (1 + deltaR*U2))

    nonzero_sini_obs = (sini_n_obs > 0) & (sini_n_obs < 2)
    sini_n_obs = sini_n_obs[nonzero_sini_obs]
    #print(np.min(sini_n_obs),np.max(sini_n_obs))

    scattered_incs_good_vsini = sini_n_obs[sim_vsini[nonzero_sini_obs] > vsini_threshold]

    if interp==True:
        if len(scattered_incs_good_vsini) != 0:
            mcincemp = emp.ECDF(scattered_incs_good_vsini)
            mcincempx1 = mcincemp.x[1:]
            mcincempy1 = mcincemp.y[1:]
        else:
            mcincempy1 = np.zeros(len(xdatapoints)) + 1
            mcincempx1 = xdatapoints

        #final_cdf_vals_y = np.interp(xdatapoints,mcincempx1,mcincempy1)
        #final_cdf_vals_x = np.interp(ydatapoints,mcincempy1,mcincempx1)
        final_cdf_vals = np.interp(ydatapoints,mcincempy1,mcincempx1)
        #return final_cdf_vals_x, final_cdf_vals_y
    else:
        final_cdf_vals = scattered_incs_good_vsini
        #return final_cdf_vals
    return final_cdf_vals


def mc_sini_distrib_uncerts_fraction(sim_vrot,alpha,lamda,f,d_lnV,sig_lnV,deltaP,deltaR,vsini_threshold=0,interp=False,xdatapoints=None,ydatapoints=None,tdistrib=False):
    n = len(sim_vrot)

    sini_n_obs_isotropic = mc_sini_distrib_new(sim_vrot,45,90,d_lnV=d_lnV,sig_lnV=sig_lnV,deltaP=deltaP,deltaR=deltaR,vsini_threshold=vsini_threshold,tdistrib=tdistrib)
    sini_n_obs_isotropic = np.random.choice(sini_n_obs_isotropic, np.int(np.round((1-f)*n)))

    sini_n_obs_aligned = mc_sini_distrib_new(sim_vrot,alpha,lamda,d_lnV=d_lnV,sig_lnV=sig_lnV,deltaP=deltaP,deltaR=deltaR,vsini_threshold=vsini_threshold,tdistrib=tdistrib)
    sini_n_obs_aligned = np.random.choice(sini_n_obs_aligned, np.int(np.round(f*n)))

    sini_n_obs = np.concatenate([sini_n_obs_isotropic, sini_n_obs_aligned])
    scattered_incs_good_vsini = sini_n_obs

    if interp==True:
        if len(scattered_incs_good_vsini) != 0:
            mcincemp = emp.ECDF(scattered_incs_good_vsini)
            mcincempx1 = mcincemp.x[1:]
            mcincempy1 = mcincemp.y[1:]
        else:
            mcincempy1 = np.zeros(len(xdatapoints)) + 1
            mcincempx1 = xdatapoints

        #final_cdf_vals = np.interp(xdatapoints,mcincempx1,mcincempy1)
        final_cdf_vals = np.interp(ydatapoints,mcincempy1,mcincempx1)
    else:
        final_cdf_vals = scattered_incs_good_vsini

    return final_cdf_vals

def obs_sini_distrib_uncerts_specific_fast(xdatapoints,alpha,lamda,deltasini=0.11641160175478454,deltap=0.038906455369088946,deltav=0.05176470588235295,deltar=0.023155991748531755
,cutoff=0.,n=50000,lognorm=False):
    #xdatapoints = np.sort(xdatapoints)
    sini_n_true = true_sini_distrib_specific(alpha,lamda,n)

    sini_n_true = sini_n_true[sini_n_true > cutoff]
    U1 = np.random.normal(size=len(sini_n_true))
    U2 = np.random.normal(size=len(sini_n_true))

    deltapv = np.sqrt(deltap**2+deltav**2)
    #print((1 + deltapv*U1)/(1 + deltar*U2))

    #sini_n_obs = sini_n_true * ((1 + deltapv*U1)/(1 + deltar*U2))
    if lognorm == False:
        sini_n_obs = sini_n_true * ((1 + deltapv*U1)/(1 + deltar*U2))
    elif lognorm == True:
        U3 = np.random.lognormal(size=len(sini_n_true))
        plusorminus = np.random.choice([-1,1],len(sini_n_true))
        #U4 = np.random.normal(size=n)
        sini_n_obs = sini_n_true * ((1 + deltap*U1) * (1 + deltav*U3*plusorminus) / (1 + deltar*U2))

    #print(xdatapoints)
    #print(sini_n_obs)

    #cutoffmask = np.zeros(len(sini_n_obs),dtype=bool)
    upperlimit=3

    keepthese = (sini_n_obs < upperlimit) & (sini_n_obs > 0)
    sini_n_obs = sini_n_obs[keepthese]

    #for i in range(len(sini_n_obs)):
        ###if sini_n_obs[i] > cutoff:
        #if (sini_n_true[i] > cutoff) & (sini_n_obs[i] < upperlimit) & (sini_n_obs[i] > 0):
        #    cutoffmask[i] = True

    #sini_n_obs = sini_n_obs[cutoffmask]
    if len(sini_n_obs) != 0:
        mcincemp = emp.ECDF(sini_n_obs)
        mcincempx1 = mcincemp.x[1:]
        mcincempy1 = mcincemp.y[1:]
    else:
        mcincempy1 = np.zeros(len(xdatapoints)) + 1
        mcincempx1 = xdatapoints

    #print(mcincempx1)

    #print(mcincempy1)

    #print(mcincempx1)
    #print(mcincempy1)
    #sini_n_obs = np.interp(xdatapoints,mcincemp.x[1:],mcincemp.y[1:])
    sini_n_obs = np.interp(xdatapoints,mcincempx1,mcincempy1)


    return sini_n_obs


def obs_sini_distrib_uncerts_fraction(xdatapoints,alpha,lamda,f,deltasini=0.11641160175478454,deltap_iso=0.038906455369088946,deltav_iso=0.05176470588235295,deltar_iso=0.023155991748531755,deltap_aniso=0.038906455369088946,deltav_aniso=0.05176470588235295,deltar_aniso=0.023155991748531755
,cutoff=0.,n=50000):

    sini_n_obs_isotropic = obs_sini_distrib_uncerts(xdatapoints,45,90,deltasini=deltasini,deltap=deltap_iso,deltav=deltav_iso,deltar=deltar_iso
    ,cutoff=cutoff,n= np.int(n * (1-f)))

    sini_n_obs_aligned = obs_sini_distrib_uncerts(xdatapoints,alpha,lamda,deltasini=deltasini,deltap=deltap_aniso,deltav=deltav_aniso,deltar=deltar_aniso
    ,cutoff=cutoff,n= np.int(n * f))

    sini_n_obs = np.concatenate([sini_n_obs_isotropic, sini_n_obs_aligned])

    return sini_n_obs

def obs_sini_distrib_uncerts_specific_fraction(xdatapoints,alpha,lamda,f,deltasini=0.11641160175478454,deltap_iso=0.038906455369088946,deltav_iso=0.05176470588235295,deltar_iso=0.023155991748531755,deltap_aniso=0.038906455369088946,deltav_aniso=0.05176470588235295,deltar_aniso=0.023155991748531755
,cutoff=0.,n=50000):

    sini_n_obs_isotropic = obs_sini_distrib_uncerts(xdatapoints,45,90,deltasini=deltasini,deltap=deltap_iso,deltav=deltav_iso,deltar=deltar_iso
    ,cutoff=cutoff,n= np.int(n * (1-f)))

    sini_n_obs_aligned = obs_sini_distrib_uncerts(xdatapoints,alpha,lamda,deltasini=deltasini,deltap=deltap_aniso,deltav=deltav_aniso,deltar=deltar_aniso
    ,cutoff=cutoff,n= np.int(n * f))

    sini_n_obs = np.concatenate([sini_n_obs_isotropic, sini_n_obs_aligned])

    if len(sini_n_obs) != 0:
        mcincemp = emp.ECDF(sini_n_obs)
        mcincempx1 = mcincemp.x[1:]
        mcincempy1 = mcincemp.y[1:]
    else:
        mcincempy1 = np.zeros(len(xdatapoints)) + 1
        mcincempx1 = xdatapoints

    #print(mcincempx1)
    #print(mcincempy1)
    #sini_n_obs = np.interp(xdatapoints,mcincemp.x[1:],mcincemp.y[1:])
    sini_n_obs = np.interp(xdatapoints,mcincempx1,mcincempy1)
    return sini_n_obs

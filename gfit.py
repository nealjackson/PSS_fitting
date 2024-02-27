import astropy, numpy as np,scipy,sys,os,emcee,pickle,matplotlib
from scipy.ndimage import maximum_position; from scipy.signal import convolve2d
from scipy.optimize import fmin
from matplotlib import pyplot as plt
from astropy.io.fits import getdata,getheader,writeto
from astropy.modeling.functional_models import Sersic2D
DSIZ, DSHAPE, VERYBIG, bands = 40, (40,40), 1.0E12, ['g','r','i','z']

# Stage 0: utility routines for elliptical moffat etc
#  elliptical moffat from aspylib.com/doc/aspylib_fitting.html
#  alpha(1+A(x-x0)**2+B*(y-y0)**2+C(x-x0)(y-y0)**2)**-gamma

def el_moffat (ishape, x_0,y_0, A, B, C, alpha, gamma):
    xx,yy = np.meshgrid(np.arange(-x_0,ishape[1]-x_0),np.arange(-y_0,ishape[0]-y_0))
    return alpha*(np.ones(ishape)+A*xx**2+B+yy**2+C*xx*yy)**(-gamma)

def sanity (p, llim, ulim):  # sanity check: chisq gets bigger if further from limit
    if p<llim:
        return VERYBIG * (llim-p)
    elif p>ulim:
        return VERYBIG * (p-ulim)
    else:
        return 0

def fit_psf (x0, *x):
    mofdata = el_moffat(DSHAPE,x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6])
    return np.sqrt(np.sum((x[0]-mofdata)**2))

# Stage 1: get Moffat profiles from psf stars

for col in bands:
    psf_data = getdata(col+'_psf.fits')
    psf_guess_pos = maximum_position(psf_data)
    psf_guess_max = psf_data[psf_guess_pos]
    guess = [psf_guess_pos[0],psf_guess_pos[1],1.0,1.0,1.0,psf_guess_max,1.0]
    xop = fmin(fit_psf,guess,args=(psf_data,),maxiter=10000)
    mofdata = el_moffat(DSHAPE,xop[0],xop[1],xop[2],xop[3],xop[4],xop[5],xop[6])
    chisq = np.sqrt(np.sum((psf_data-mofdata)**2))
    new = xop[2:7]   # A,B,C,scaling,gamma - scaling to be ignored
    try:
        moffat_parms = np.vstack((moffat_parms,new))
    except:
        moffat_parms = np.copy(xop[2:7])

# Stage 2: read initial parameters (hard-wired) - hard-wired constraint
# on galaxy not being outside the rectangle defined by the images as well

def fit_img (x0, *x):
    xP1,yP1,xP2,yP2,xG,yG,AP1G,AP2G,AGG,AP1R,AP2R,AGR,AP1I,AP2I,AGI,\
        AP1Z,AP2Z,AGZ,greff,gn,gellip,gtheta = x0
    data, mo, dumpmodel, ismcmc = x 
    bands = ['g','r','i','z']
    chisq = 0.0
    amp1,amp2,ampg = [AP1G,AP1R,AP1I,AP1Z],[AP2G,AP2R,AP2I,AP2Z],[AGG,AGR,AGI,AGZ]
    if not ismcmc:
        chisq = sanity (xP1,0,DSIZ)+sanity (yP1,0,DSIZ)+sanity (xP2,0,DSIZ) +\
            sanity (yP2,0,DSIZ)+sanity (xG,17,23)+sanity (yG,17,26)
        for param in amp1+amp2+ampg:
            chisq += sanity (param, 0, VERYBIG)
    if chisq!=0:   # failed sanity check somewhere
        return chisq
    for tb in bands:
        nb = bands.index(tb)
        sersic = Sersic2D (amplitude=ampg[nb],r_eff=greff,n=gn,ellip=gellip,\
                           theta=gtheta,x_0=xG,y_0=yG)
        sersic_im = sersic.render(np.zeros_like(data[2*nb]))
        m0,m1,m2,m4 = mo[nb,0],mo[nb,1],mo[nb,2],mo[nb,4]
        moffat = el_moffat (DSHAPE, DSIZ/2, DSIZ/2, m0, m1, m2, 1, m4)
        model1 = el_moffat (DSHAPE,xP1,yP1, m0, m1, m2, amp1[nb], m4)
        model2 = el_moffat (DSHAPE,xP2,yP2, m0, m1, m2, amp2[nb], m4)
        modelg = convolve2d (sersic_im,moffat,mode='same')
        model = model1+model2+modelg
        nchisq = np.sum((model-data[2*nb])**2/data[2*nb+1])
        chisq += nchisq
        if dumpmodel:
            writeto('%s_mod.fits'%tb,model,overwrite=True)
    if not ismcmc:
        for i in range(len(x0)):
            sys.stdout.write('%.2f '%x0[i])
        print('%.1f'% (chisq))
    if ismcmc and np.isnan(chisq):
        return -VERYBIG
    return -chisq/2.0 if ismcmc else chisq

data = np.zeros((8,DSIZ,DSIZ))   # data stack is G GWT R RWT I IWT Z ZWT
for i in range(4):
    data[2*i] = getdata(bands[i]+'.fits')
    data[2*i+1] = getdata(bands[i]+'_wt.fits')

# some hardwired guesses for this case (J0011-0845)
xP1,yP1,xP2,yP2,xG,yG,AP1G,AP2G,AGG,AP1R,AP2R,AGR,AP1I,AP2I,AGI,\
     AP1Z,AP2Z,AGZ,greff,gn,gellip,gtheta = \
     18.04,17.89,21.25,24.32,19.8,19.5,\
     7.89e6,8.32e6,0.01, 43.35e6, 36.27e6,123.72e6,\
     4.57e6,3.87e6,12.64e6, 7.53e6,4.55e6,15.46e6, \
     0.13,3.61,0.09,0.27
guess = [xP1,yP1,xP2,yP2,xG,yG,AP1G,AP2G,AGG,AP1R,AP2R,AGR,AP1I,AP2I,AGI,\
        AP1Z,AP2Z,AGZ,greff,gn,gellip,gtheta]
# run twice, once to do the optimization and once to dump out the final model as model.fits
xop = fmin(fit_img,guess,args=(data,moffat_parms,False,False),maxiter=1000)
junk = fmin(fit_img,xop,args=(data,moffat_parms,True,False),maxiter=2)

for i in range(len(xop)):
    sys.stdout.write('%.2f '%xop[i])
sys.stdout.write('\n')

# Stage 3: make plots of the data/residuals

DSIZ, DSHAPE, VERYBIG, bands = 40, (40,40), 1.0E12, ['g','r','i','z']
for tb in bands:
    nb = bands.index(tb)
    tdata = getdata('%s.fits'%tb)
    tmodel = getdata('%s_mod.fits'%tb)
    plt.subplots_adjust(wspace=0.01,hspace=0.01,left=0.0,right=1.0,top=1.0,bottom=0.0)
    plt.subplot(3,len(bands),nb+1,xticks=[],yticks=[]);plt.imshow(tdata)
    tmax = tdata.max()
    plt.subplot(3,len(bands),nb+1+len(bands),xticks=[],yticks=[]);plt.imshow(tmodel,vmax=tmax)
    plt.subplot(3,len(bands),nb+1+len(bands)*2,xticks=[],yticks=[]);plt.imshow(tdata-tmodel,vmax=tmax)
    os.system('rm model.png');plt.savefig('model.png',bbox_inches='tight')

f = open('model.txt','w'); f.write('Model parameters\n')
for i in range(len(xop)):
    f.write('%2f '%xop[i])
f.write('\n'); f.close()

# Stage 4: use the optimized parameters for an MCMC

nwalkers=60; nburn=500; niter=500
smc = emcee.EnsembleSampler(nwalkers,len(xop),fit_img,args=(data,moffat_parms,False,True))
xvar = np.array([1,1,1,1,1,1,1e6,1e6,1000,1e6,1e6,1000,1e6,1e6,1000,1e6,1e6,1000,.1,.1,.1,.1])
p0 = [xop+xvar*(0.5-np.random.rand(len(xop))) for i in range(nwalkers)]
pos,prob,state = smc.run_mcmc(p0,nburn)
os.system('rm mcmc_Bchain.npy');np.save('mcmc_Bchain',smc.chain)
smc.reset()
smc.run_mcmc(pos,niter,rstate0=state)
os.system('rm mcmc_chain.npy');np.save('mcmc_chain',smc.chain)


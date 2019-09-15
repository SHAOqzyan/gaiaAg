
import numpy as np
from scipy.stats import norm
from scipy  import special
from progressbar import *



nominator=np.sqrt(2) #*sigma
logsqrt2pi=0.5*np.log(2*np.pi)
agmin= 0 #np.min(dataAG)
agmax= 3.609 #np.max(dataAG)
sqrtPi2i=2./np.sqrt(2*np.pi)


logsqrtPi2i=np.log(sqrtPi2i)

##
# Use emcee to sove distance
# Note: mu1 is set to 0.



######################$$$$$$$$$$$$$
#p , e = optimize.curve_fit(piecewise_linear, x, y)
def lnprior(theta,minDis,maxDis):
	disCloud, imu1sigma,mu2,imu2sigma=theta
	#if imu1sigma<0 or  imu2sigma<0 or mu1<0 or mu2<0 or mu1>3.609 or mu2>3.609:
	if imu1sigma<0.15 or imu1sigma>10  or imu2sigma<0.15 or imu2sigma>10 or   mu2<0.0   or mu2>3.609 or disCloud<minDis or disCloud>maxDis-50:

		return -np.inf
	return 0.0
def calProbEmcee(theta,dataDis, dataAG,disError, AGError,AGErrorSquare,Nstars,minDis,maxDis):
	##calculate the probablitydisSigma
	#trueError=np.sqrt(disError**2+disSigma**2)
	#trueError= #np.sqrt(disError**2+disSigma**2)
	lp=lnprior(theta,minDis,maxDis)
	if not np.isfinite(lp):
		return -np.inf
	disCloud,  imu1sigma,mu2,imu2sigma=theta
	mu1=0
	weight=  norm.cdf(disCloud,dataDis,disError)

	#weight=1-scipy.stats.norm.cdf(dataDis,disCloud, trueError)

	#fore ground
	errorVar= 1./imu1sigma**2+AGErrorSquare
	errorVar_i=1./errorVar
	convolveSTDi=  np.sqrt(errorVar_i)
	a=  special.erf((agmax-mu1)/nominator*convolveSTDi)+ special.erf((mu1-agmin)/nominator*convolveSTDi)

	#common factor,  np.exp(-0.5*(dataAG-mu1)**2/errorVar)*convolveSTDi*sqrtPi2i/a

	#pForeground= np.exp(-0.5*(dataAG-mu1)**2/errorVar)*convolveSTDi*sqrtPi2i/a
	pForeground= np.exp(-0.5*np.square(dataAG-mu1)*errorVar_i)*convolveSTDi/a


	errorVar2= 1./imu2sigma**2 +AGErrorSquare
	errorVar2_i=1./errorVar2
	convolveSTDi2=  np.sqrt(errorVar2_i)
	a2=  special.erf((agmax-mu2)/nominator*convolveSTDi2)+ special.erf((mu2-agmin)/nominator*convolveSTDi2)


	pBackground= np.exp(-0.5*np.square(dataAG-mu2)*errorVar2_i)*convolveSTDi2/a2

	totalP= pBackground+(pForeground-pBackground)*weight #true norm

	return np.sum(np.log(totalP) )+Nstars*logsqrtPi2i



def testEmcee(dataDis, dataAG,disError, AGError,AGErrorSquare,Nstars ):
	###try emcee
	print "testing Emcee"
	import emcee

	minDis= np.min(dataDis)

	maxDis= np.max(dataDis)  #to avoid touch the edge

	last100pcAG=dataAG[dataDis>maxDis-50]
	mu20=np.mean(last100pcAG)
	isigma0=np.std(last100pcAG,ddof=1)


	isigma0= 1./isigma0


	ndim = 4
	nwalkers=10
	np.random.seed()
	N=1000

	p0=[]


	for i in range(nwalkers):
		#a= np.array([np.random.uniform(minDis,maxDis),np.random.exponential(0.5),np.random.exponential(2),np.random.exponential(1),np.random.exponential(2)] )
		a= np.array([np.random.uniform(minDis,maxDis-50), np.random.exponential(2),np.random.exponential(mu20),np.random.exponential(isigma0)] )

		p0.append(a)

	sampler = emcee.EnsembleSampler(nwalkers, ndim, calProbEmcee, args=[dataDis, dataAG,disError, AGError,AGErrorSquare,Nstars,minDis,maxDis])
	pos, prob, state = sampler.run_mcmc(p0, 200)
	sampler.reset()

	sampler.run_mcmc(pos, N)
	for i in range(ndim):

		print "sampleNumbers:",sampler.flatchain[:,i].shape
		print "Mean values:",np.mean(sampler.flatchain[:,i])

	#returnSampleDic[processID]= [sampler.flatchain[:,0][::15], 1./sampler.flatchain[:,1][::15],sampler.flatchain[:,2][::15],1./sampler.flatchain[:,3][::15]]

	print "Emcee done"
	return [sampler.flatchain[:,0][::10], 1./sampler.flatchain[:,1][::10],sampler.flatchain[:,2][::10],1./sampler.flatchain[:,3][::10]]



######################$$$$$$$$$$$$$




if 1:

    dis=np.load("testStarDis.npy")
    disStd=np.load("testStarDisErr.npy")
    Ag=np.load("testStarAg.npy")
    AgStd=np.load("testStarAgErr.npy")

    Nstars=  len(dis )   #,dataAGSquare):

    AGErrorSquare= AgStd*AgStd
    testEmcee(dis, Ag,disStd, AgStd,AGErrorSquare,Nstars )
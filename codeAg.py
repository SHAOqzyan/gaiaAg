
import numpy as np
from scipy.stats import norm
from scipy  import special
from progressbar import * 

####
#Author Qing-Zeng Yan
#An example of MCMC code to calculate distances with Gaia DR2 Ag
####


#constants
nominator=np.sqrt(2) #*sigma
logsqrt2pi=0.5*np.log(2*np.pi)
agmin= 0 #np.min(dataAG)
agmax= 3.609 #np.max(dataAG)
sqrtPi2i=2./np.sqrt(2*np.pi) 


logsqrtPi2i=np.log(sqrtPi2i)


def getNextValues( runSamples,indexPara):
	
	RMSs=[50,0.3,0.3,0.3,0.3]
	currentValues=[  ]

	for j in range(len(runSamples)):
		currentValues.append(runSamples[j][-1])
		
		 
	
	currentValues[indexPara]=currentValues[indexPara] +np.random.normal(0,RMSs[indexPara])
	
	return currentValues



def calProb(disCloud,  mu1,imu1sigma,mu2,imu2sigma,dataDis, dataAG,disError, AGError,AGErrorSquare,Nstars):
	"""
	# PDF of truncated gaussian
	:param disCloud: 
	:param mu1: 
	:param imu1sigma: 
	:param mu2: 
	:param imu2sigma: 
	:param dataDis: 
	:param dataAG: 
	:param disError: 
	:param AGError: 
	:param AGErrorSquare: # square of Ag, to improve the speed of calculation
	:param Nstars: #total number of on-cloud stars
	:return: 
	"""


	weight= norm.cdf(disCloud,dataDis,disError)


	errorVar= 1./imu1sigma**2+AGErrorSquare
	errorVar_i=1./errorVar
	convolveSTDi=  np.sqrt(errorVar_i)
	a=  special.erf((agmax-mu1)/nominator*convolveSTDi)+ special.erf((mu1-agmin)/nominator*convolveSTDi)


	pForeground= np.exp(-0.5*np.square(dataAG-mu1)*errorVar_i)*convolveSTDi/a


	errorVar2= 1./imu2sigma**2 +AGErrorSquare
	errorVar2_i=1./errorVar2
	convolveSTDi2=  np.sqrt(errorVar2_i)
	a2=  special.erf((agmax-mu2)/nominator*convolveSTDi2)+ special.erf((mu2-agmin)/nominator*convolveSTDi2)


	pBackground= np.exp(-0.5*np.square(dataAG-mu2)*errorVar2_i)*convolveSTDi2/a2

	totalP= pBackground+(pForeground-pBackground)*weight #true norm

	return np.sum(np.log(totalP) )+Nstars*logsqrtPi2i

def getDisAndErrorMCMCTrucatedGau5p(  dataDis, dataAG,disError, AGError,  mu1=0,distance0=-1,maxDistanceSearch=1000,distanceError=200,sampleN=1000,burn_in=50):

	"""
	do MCMC in this single function
	"""
	#first


	AGErrorSquare=AGError*AGError

	Nstars=  len(dataAG )*1.0  #,dataAGSquare):
	np.random.seed()
	#print "Calculating distances with MCMC...total smaple number is {}....".format(sampleN)

	minDis=int(np.min(dataDis))

	maxDis=int(np.max(dataDis))  #to avoid touch the edge

	last100pcAG=dataAG[dataDis>maxDis-50]
	mu20=np.mean(last100pcAG)
	isigma0=np.std(last100pcAG,ddof=1)




	isigma0= 1./isigma0


	aaa=0.5*np.log(2*np.pi)




	disList=[]
	mu1List=[]
	mu2List=[]
	imu1sigmaList=[]
	imu2sigmaList=[]


	disCloud=np.random.uniform(minDis,maxDis-50) # maxDistanceSearch) #np.random.normal(distance0,distanceError) # #choice(pns)
	mu2=  np.random.exponential(mu20)   # np.random.uniform(0,3.609)  #abs( np.random.normal(mu20,0.45) )
 	imu1sigma=  np.random.exponential(2)
	imu2sigma=  np.random.exponential(isigma0)
	mu1=  np.random.exponential(0.5)

 	p0=calProb(disCloud, mu1,imu1sigma,mu2,imu2sigma,dataDis, dataAG,disError, AGError,AGErrorSquare,Nstars)

	if np.isnan(p0):
		print "Wrong intial value!!!!!!!!!!!!!!!!!!!!!!!!!!"
		for i in range(1000):
			disCloud=np.random.uniform(minDis,maxDis) # maxDistanceSearch) #np.random.normal(distance0,distanceError) # #choice(pns)
			mu1=  np.random.exponential(0.5)

			mu2=  np.random.exponential(mu20)   # np.random.uniform(0,3.609)  #abs( np.random.normal(mu20,0.45) )
		 	imu1sigma=  np.random.exponential(2)
			imu2sigma=  np.random.exponential(isigma0)
 			p0=calProb(disCloud,  mu1,imu1sigma,mu2,imu2sigma,dataDis, dataAG,disError, AGError,AGErrorSquare,Nstars)
			if not np.isnan(p0):
				break

	runSamples= [[disCloud],[mu1],[imu1sigma],[mu2], [imu2sigma] ]

	widgets = ['MCMCSmapleDistance: ', Percentage(), ' ', Bar(marker='>',left='|',right='|'),
	           ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

	pbar = ProgressBar(widgets=widgets, maxval=sampleN+burn_in+1)
	pbar.start()

	searchMax=maxDis-50


	recordinverval=15

	for i in range(10000000):
		if i>5000 and np.mean(disList)==disList[-1] :

			print p0,p1,"Wrong, restart"
			print "Parameter values:",disCloud, mu1,imu1sigma,mu2,imu2sigma

			getDisAndErrorMCMCTrucatedGau(  dataDis, dataAG,disError, AGError, processID, returnSampleDic,  mu1=mu1,distance0=distance0,maxDistanceSearch=maxDistanceSearch,distanceError=distanceError,sampleN=sampleN,burn_in=burn_in)
			return

		paraJ=0
		while paraJ<5:
			valueCand= getNextValues(runSamples,paraJ)
			disCloud,  mu1,imu1sigma,mu2,imu2sigma=valueCand

			if imu1sigma <0 or imu2sigma<0 or mu2<0 or disCloud<minDis or disCloud>searchMax or mu1<0:
				# constraints
				continue
			p1=	 calProb(disCloud, mu1,imu1sigma,mu2,imu2sigma,dataDis, dataAG,disError, AGError,AGErrorSquare,Nstars)

			randomR=np.random.uniform(0,1)



			if p1>=p0 or p1-p0>np.log(randomR):
				#print disCloud,  imu1sigma,mu2,imu2sigma,"--->",processID

				p0=p1;
				runSamples[paraJ].append( valueCand[paraJ] )

			else:
				runSamples[paraJ].append( runSamples[paraJ][-1] )
			paraJ=paraJ+1


		#record the last value of samples
		runSamples=[ runSamples[0][-1:],runSamples[1][-1:], runSamples[2][-1:], runSamples[3][-1:],runSamples[4][-1:] ]



		if i%recordinverval==0:


			disList.append(runSamples[0][-1] )
			mu1List.append(runSamples[1][-1] )

			imu1sigmaList.append(runSamples[2][-1]  )
			mu2List.append(runSamples[3][-1] )
			imu2sigmaList.append(runSamples[4][-1]  )




			pbar.update(len(disList)) #this adds a little symbol at each iteration
			if len(disList)>burn_in+sampleN:

				break



	pbar.finish()


	print "The accept rate is", len(disList)*4./i
	print "i is ",i

	if 1: #test normal samplig

		disArray=np.array(disList[burn_in:-1])

		#for mu1 t
		mu1Array=np.array(mu1List[burn_in:-1])
		#print "The modeled mu1 is ",np.mean(mu1ArrayT)

 
		mu2Array=np.array(mu2List[burn_in:-1])

		mu1SigmaArray=1./np.array( imu1sigmaList[burn_in:-1])
		mu2SigmaArray=1./np.array( imu2sigmaList[burn_in:-1])


		#print "Testing correlation time"
		#calAuto(disArray)
		#calAuto(mu1Array)
		#calAuto(mu1SigmaArray)
		#calAuto(mu2Array)
		#calAuto(mu2SigmaArray)

		return [disArray, mu1Array, mu1SigmaArray,mu2Array,mu2SigmaArray]



if 1:

	#read data, of cloud MBMB54

	dis=np.load("testStarDis.npy")
	disStd=np.load("testStarDisErr.npy")
	Ag=np.load("testStarAg.npy")
	AgStd=np.load("testStarAgErr.npy")


	samples=getDisAndErrorMCMCTrucatedGau5p(  dis, Ag,disStd, AgStd )
	print "The modelled distance  is {}".format( np.mean(samples[0]))


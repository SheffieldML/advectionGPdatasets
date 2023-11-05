import pickle
from advectionGPdatasets.roundhill import RoundHill,RoundHillModel
from advectionGP.kernels import EQ
import numpy as np
import matplotlib.pyplot as plt
import argparse
#parameters 

Nfeat = 2000
Npart = 30
k_0 = 0.2
res = [200,100,100]
ls = 2.0
s2 = 1000


rhms = []
plt.figure()
for rep in np.arange(10):
    try:
        rhm = pickle.load(open('roundhillmodel_noholdout_s2=%d_ls=%d_k0=0.5_Nfeat=%d_Nparticles=%d_rep=%d.pkl' % (s2,ls,Nfeat,Npart,rep),'rb'))
    except:
        continue
    rhms.append(rhm)
    for sample_at_peak,col,style,lab in zip([True,False],['b','g'],['-','--'],['around peak','whole domain']):
        sourcemasses = []        
        if sample_at_peak:
            sources = rhm.results['sources']['all'][0,:,35:50,0:20].copy()
        else:
            sources = rhm.results['sources']['all'][0,:,:,:].copy()
        mass = np.sum(sources,(1,2))*np.prod(rhm.mInfer.getGridStepSize()[0][1:])

        sourcemasses.append(mass)
        times = np.arange(rhm.mInfer.boundary[0][0],rhm.mInfer.boundary[1][0],rhm.mInfer.getGridStepSize()[0][0])[:-1]
        m = np.mean(sourcemasses,0)[:-1]/1e3 #grams from mg.
        #s = (np.std(sourcemasses,0)/np.sqrt(13))[:-1]/1e3 #13 samples (finding rough SE! argh)
        #sourcemasses = np.array(sourcemasses)
        plt.plot(times,m,style+col,label=lab)
        #plt.fill_between(times,m-s*1.96,m+s*1.96,color=col,alpha=0.2)
        plt.ylabel('Mass flow rate / $g\;s^{-1}$')
        plt.xlabel('Time / $s$')
        #plt.ylim([-5,12.5])#12.500])
        plt.xlim(rhm.mInfer.boundary[0][0],rhm.mInfer.boundary[1][0]-4)
plt.grid()
plt.hlines(0,-200,1000)
plt.legend()
plt.ylim([-3,3])
plt.savefig('estimated_flowrate_s2=%d_ls=%d_k0=0.5_Nfeat=%d_Nparticles=%d_rep=%d.pdf' % (s2,ls,Nfeat,Npart,rep))
 	

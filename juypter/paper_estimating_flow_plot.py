import pickle
from advectionGPdatasets.roundhill import RoundHill,RoundHillModel
from advectionGP.kernels import EQ
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='Run roundhill pollution model')
parser.add_argument("--features", help="Number of features", type=int)
parser.add_argument("--particles", help="Number of particles per observation", type=int)
parser.add_argument("--ls", help="Lengthscale", type=float)
parser.add_argument("--k0", help="k0", type=float)
parser.add_argument("--resT", help="resolution (in time)", type=int)
parser.add_argument('--ground', default=False, action='store_true', help='include reflecting ground plane')
args = parser.parse_args()

rhms = []
plt.figure()
def getfn(walls,s2,ls,k0,Nfeat,Npart,rep,fig):
    if fig:
        fn = 'roundhillmodel3d_%s_noholdout_s2=%d_ls=%d_k0=%0.1f_Nfeat=%d_Nparticles=%d.pdf' % ('wall' if walls is not None else '',s2,ls,k0,Nfeat,Npart)
    else:
        fn = 'roundhillmodel3d_%s_noholdout_s2=%d_ls=%d_k0=%0.1f_Nfeat=%d_Nparticles=%d_rep=%d.pkl' % ('wall' if walls is not None else '',s2,ls,k0,Nfeat,Npart,rep)    
    return fn

for rep in range(20):
    fn = getfn(walls,s2,ls,k0,Nfeat,Npart,rep)
    print("Loading %s: " % fn,end="")
    try:
        rhm = pickle.load(open(fn,'rb'))
        print("Success")
    except:
        print("Failed")
        continue
    rhms.append(rhm)

for rhm in rhms:
    for sample_at_peak,col,style,lab in zip([True,False],['b','g'],['-','--'],['around peak','whole domain']):
        sourcemasses = []        
        if sample_at_peak:
            sources = rhm.results['sources']['all'][0,:,(73-15):(73+15),(12-12):(12+20),:3].copy()
        else:
            sources = rhm.results['sources']['all'][0,:,:,:,:].copy()
        pickle.dump(sources,open('temp_'+lab+'.pkl','wb'))
        #sources[sources<0]=0
        #np.sum(sources)*np.prod(rhm.mInfer.getGridStepSize()[0][1:])
        mass = np.sum(sources,(1,2,3))*np.prod(rhm.mInfer.getGridStepSize()[0][1:])

        sourcemasses.append(mass)
        #plt.plot(times,mass[:-1])
        times = np.arange(rhm.mInfer.boundary[0][0],rhm.mInfer.boundary[1][0],rhm.mInfer.getGridStepSize()[0][0])[:-1]
        m = np.mean(sourcemasses,0)[:-1]/1e3 #grams from mg.
        #s = (np.std(sourcemasses,0)/np.sqrt(13))[:-1]/1e3 #13 samples (finding rough SE! argh)
        #s = np.sqrt((np.var(sourcemasses,0)/13+single_sample_var))[:-1]/1e3
        #s = np.sqrt(single_sample_var)[:-1]/1e3
        #sourcemasses = np.array(sourcemasses)
        plt.plot(times,m,style+col,label=lab)
        #plt.fill_between(times,m-s*1.96,m+s*1.96,color=col,alpha=0.2)
        plt.ylabel('Mass flow rate / $g\;s^{-1}$')
        plt.xlabel('Time / $s$')
        #plt.ylim([-5,12.5])#12.500])
        plt.xlim(rhm.mInfer.boundary[0][0],rhm.mInfer.boundary[1][0]-4)

#plt.title("%0.3f" % k0)
#plt.hlines(39.5e3,0,300,'r')
plt.grid()
plt.hlines(0,-200,1000)
plt.legend()
#plt.ylim([-3,3])
fn = getfn(walls,s2,ls,k0,Nfeat,Npart,fig=True)
plt.savefig(fn)

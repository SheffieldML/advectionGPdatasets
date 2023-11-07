import pickle
from advectionGPdatasets.roundhill import RoundHill,RoundHillModel
from advectionGP.kernels import EQ
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='Run roundhill pollution model')
parser.add_argument("--rep", help="Starting with this random seed", type=int)
parser.add_argument("--features", help="Number of features", type=int)
parser.add_argument("--particles", help="Number of particles per observation", type=int)
parser.add_argument("--ls", help="Lengthscale", type=float)
parser.add_argument("--k0", help="k0", type=float)
parser.add_argument("--resT", help="resolution (in time)", type=int)
parser.add_argument('--ground', default=False, action='store_true', help='include reflecting ground plane')
args = parser.parse_args()
rep = args.rep
np.random.seed(rep)
Nfeat = args.features
Npart = args.particles
res = [args.resT,100,100,10]
ls = args.ls
s2 = 1000
k0 = args.k0 #0.2 #0.5
if args.ground:
    walls = [(3,0,-1)]
else:
    walls = None
    
def getfn(walls,s2,ls,k0,Nfeat,Npart,rep):
    fn = 'roundhillmodel3d_%s_noholdout_s2=%d_ls=%d_k0=%0.1f_Nfeat=%d_Nparticles=%d_rep=%d.pkl' % ('wall' if walls is not None else '',s2,ls,k0,Nfeat,Npart,rep)
    return fn

fn = getfn(walls,s2,ls,k0,Nfeat,Npart,rep)
print(fn)
try:        
    pickle.load(open(fn,'rb'))
except:        
    k = EQ(np.array([400,ls,ls,ls]), s2)
    rhm = RoundHillModel(N_feat=Nfeat,Nparticles=Npart,k=k,res=res,k_0=k0,holdout=False,walls=walls)
    
    rhm.compute(1)

    #make file smaller...
    #pickle.dump(rhm,open(fn,'wb'))
    sourcesall = rhm.results['sources']['all']
    gridstepsize = rhm.mInfer.getGridStepSize()[0]
    boundary = rhm.mInfer.boundary
    print("Saving to %s" % ('small_'+fn))
    pickle.dump({'sourcesall':sourcesall,'gridstepsize':gridstepsize,'boundary':boundary},open('small_'+fn,'wb'))


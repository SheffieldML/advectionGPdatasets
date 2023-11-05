import pickle
from advectionGPdatasets.roundhill import RoundHill,RoundHillModel
from advectionGP.kernels import EQ
import numpy as np
import matplotlib.pyplot as plt
import argparse
#parameters 
parser = argparse.ArgumentParser(description='Run roundhill pollution model')
parser.add_argument("rep", help="Starting with this random seed", type=int)

args = parser.parse_args()


np.random.seed(args.rep)

Nfeat = 2000
Npart = 30
k_0 = 0.2
res = [200,100,100]
ls = 2.0
s2 = 1000

fn = 'roundhillmodel_noholdout_s2=%d_ls=%d_k0=%0.1f_Nfeat=%d_Nparticles=%d_rep=%d.pkl' % (s2,ls,k_0,Nfeat,Npart,rep)
try:        
    pickle.load(open(fn,'rb'))
except:
    k = EQ(np.array([400,ls,ls]), s2)
    rhm = RoundHillModel(N_feat=Nfeat,Nparticles=Npart,k=k,res=res,k_0=k_0,holdout=False)
    rhm.compute(1)
    pickle.dump(rhm,open(fn,'wb'))

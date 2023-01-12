from advectionGPdatasets.roundhill import RoundHill,RoundHillModel
from advectionGP.kernels import EQ
import pickle
import sys
import numpy as np
for Nfeat in [10000]:
    for s2 in [200,1000,40]:
        for tls in [200,50,400]:
            for ls in [float(sys.argv[1])]: #[0.1,0.25,0.5,2,1,20]:
                for k_0 in [0.1,0.25,0.5,1,2,4,8,0.05,16,0.025,32]:
                    filename = 'roundhillmodel_s2=%d_tls=%d_ls=%0.2f_k0=%0.2f_Nfeat=%d.pkl' % (s2,tls,ls,k_0,Nfeat)
                    print("Running model. Will write to %s" % filename)
                    k = EQ(np.array([tls,ls,ls]), s2)
                    rhm = RoundHillModel(N_feat=Nfeat,Nparticles=30,k=k)
                    results = rhm.compute(30)
                    pickle.dump(rhm,open(filename,'wb'))

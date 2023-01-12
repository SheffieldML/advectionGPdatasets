from advectionGPdatasets.roundhill import RoundHill,RoundHillModel
from advectionGP.kernels import EQ

for s2 in [200,1000,40]:
    for tls in [200,50,400]:
        for ls in [9,6,3,12,2,1,20]:
            for k_0 in [0.1,0.25,0.5,1,2,4,8]:
                k = EQ(np.array([tls,ls,ls]), s2)
                filename = 'roundhillmodel_s2=%d_tls=%d_ls=%d_k0=%0.2f.pkl' % (s2,tls,ls,k_0)
                rhm = RoundHillModel(N_feat=10000,Nparticles=30,k=k)
                results = rhm.compute(30)
                pickle.dump(rhm,open(filename,'wb'))

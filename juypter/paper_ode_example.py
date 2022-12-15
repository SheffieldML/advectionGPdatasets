import numpy as np
import pickle
from advectionGP.constraints import NonNegConstraint
from advectionGP.sensors import FixedSensorModel 
from advectionGP.kernels import meshgridndim
from advectionGP.kernels import EQ 
from advectionGPdatasets.ode_modelsample import ODEModelSample
from advectionGP.models.mesh_1d_ode_model import AdjointSecondOrderODEModel as ODEModel

seed = 2
np.random.seed(seed)
#k_0 = -p0
#u   = p1 
#eta = p2
ds = ODEModelSample(non_neg=True,k_0=-0.01,u=0.2,eta=0.1,ls=0.5,Npoints=30)
X, Y, sourceGT, concTrain, boundary, m = ds.X, ds.Y, ds.source, ds.conc, ds.boundary, ds.m

pickle.dump([ds, X, Y, sourceGT, concTrain, boundary, m],open('synthetic_ODE_example_%d.pkl' % seed,'wb'))


Nsamples = 20
results = []
record = {}
ls = 0.5

uvals = np.logspace(-1.3,1.605,30)[:-1] #just removed last one!
uvals = np.r_[uvals[0::3],uvals[1::3],uvals[2::3]]
print(uvals)

for it,u in enumerate(uvals): #enumerate(np.logspace(-1.5,1/3,23)):
#for it,ls in enumerate([0.5]): #enumerate(np.logspace(-1.1,1,10)):
    print("================%0.1f [%0.5f]===============" % (it,u)) 
    np.random.seed(42)
    N_feat = 500 #Number of features used to infer the source
    k = EQ(ls, 2.0)
    res = [400]
    noiseSD = 0.1
    sensors = FixedSensorModel(X[::2],0.1)

    k_0=-0.01
    #u=0.1
    eta=0.1
    
    mTest = ODEModel(resolution=res,boundary=boundary,N_feat=N_feat,noiseSD=noiseSD,kernel=k,sensormodel=sensors,k_0=k_0,u=u,eta=eta)
    regressors = mTest.computeModelRegressors() # Compute regressor matrix
    meanZ, covZ = mTest.computeZDistribution(Y[::2]) # Infers z vector mean and covariance
    
    
    concInferred_samples = []
    sourceInferred_samples = []
    sse_samples = []
    obs_samples = []
    
    mTest.sensormodel = FixedSensorModel(X[1::2],0.1)
    for jitterA in [0,1e-8,1e-7,1e-6,1e-5,1e-4,None]:
        if jitterA is None:
            break
        try:
            Zsamps = np.random.multivariate_normal(meanZ,covZ+np.eye(len(covZ))*jitterA,Nsamples)
            break
        except np.linalg.LinAlgError:
            print("Failed to invert during Non-negative sampling. Adding jitter...")
         
    
    #we just give up with this data point...
    if jitterA is None:
        print("Giving up with this configuration")
        continue
        
    for Z in Zsamps:
        sourceInferred = mTest.computeSourceFromPhi(Z) # Generates estimated source using inferred distributio
        sourceInferred_samples.append(sourceInferred)
        
        conc = mTest.computeResponse(sourceInferred)
        concInferred_samples.append(conc)
        mTest.conc = conc
        obs = mTest.computeObservations()
        #sse_samples.append(np.sum((obs-Y[1::2])**2))
        sse_samples.append(np.sum((np.mean(np.array(obs),0)-Y[1::2])**2))
        obs_samples.append(obs)
        
    ###Non-neg calculation...
    #################################################################################################
    mTest.sensormodel = FixedSensorModel(X[::2],0.1)
    concInferred_nonneg_samples = []
    sourceInferred_nonneg_samples = []
    sse_nonneg_samples = []
    obs_nonneg_samples = []
    
    for jitterB in [0,1e-8,1e-7,1e-6,1e-5,1e-4,None]:
        if jitterB is None:
            break
        try:
            nnc = NonNegConstraint(mTest,Y[::2],np.linspace(0,20,200)[:,None],thinning=10+int(20*u),burnin=50,jitter=jitterB,verbose=True)
            break
        except np.linalg.LinAlgError:
            print("Failed to invert during Non-negative sampling. Adding jitter...")

    if jitterB is None:
        print("Giving up with this configuration")
        continue

    try:
        samps = nnc.sample(Nsamples=Nsamples)
    except NoValidStartPointFoundError:
        print("NoValidStartPointFoundError!!")
        continue
    mTest.sensormodel = FixedSensorModel(X[1::2],0.1)
    for samp in samps:
        sourceInferred = mTest.computeSourceFromPhi(samp)
        sourceInferred_nonneg_samples.append(sourceInferred)

        conc = mTest.computeResponse(sourceInferred)
        concInferred_nonneg_samples.append(conc)
        mTest.conc = conc
        obs = mTest.computeObservations()
        #sse_nonneg_samples.append(np.sum((obs-Y[1::2])**2))
        sse_nonneg_samples.append(np.sum((np.mean(np.array(obs),0)-Y[1::2])**2))
        obs_nonneg_samples.append(obs)

        
    results.append([u,np.mean(sse_nonneg_samples), np.mean(sse_samples),jitterA,jitterB])
    record[u]=[mTest,regressors,meanZ,covZ,concInferred_samples,sourceInferred_samples,sse_samples,obs_samples,concInferred_nonneg_samples,sourceInferred_nonneg_samples,sse_nonneg_samples,obs_nonneg_samples]

    pickle.dump(record,open('recordODEfindparams_u_%d_seed=%d.pkl' % (it,seed),'wb'))

import numpy as np
from advectionGP.sensors import FixedSensorModel 
from advectionGP.kernels import EQ 
from advectionGP.wind import WindSimple
from advectionGP.constraints import NonNegConstraint
from advectionGP.kernels import meshgridndim
from advectionGP.models.mesh_adr2d_model import AdjointAdvectionDiffusionReaction2DModel as PDEModel

class ModelSample():
    def __init__(self,ls=1,non_neg=False,N_feat=300):
        """
        Generates a sample from our model (with different hyperparameters). Option available to restrict it to non-negative samples
        """
        tlocL = np.linspace(1,9,10) # lower time
        xloc=np.linspace(1,9,5) # x locations
        yloc=np.linspace(1,9,5) # y locations
        sensN = len(xloc)*len(yloc) # total number of sensors 
        obsN = len(tlocL) # total time points at which an observation is taken
        X= np.zeros((obsN*sensN,4)) # obsN*sensN is total observations over all sensors and all times
        # Build sensor locations
        X[:,0] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[0] #lower time
        X[:,2] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[1] # x location
        X[:,3] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[2] # ylocation
        X[:,1] = X[:,0]+0.1 # upper time

        sensors = FixedSensorModel(X,0.1) # establish sensor model arguments are sensor locations and spatial averaging

        k_0 = 0.001 #Diffusion
        R=0
        noiseSD = 0.05 #Observation noise
        
        boundary = ([0,0,0],[10,10,10])# corners of the grid - in units of space
        k = EQ(ls, 2.0) # generate EQ kernel arguments are lengthscale and variance
        res = [200,80,80] # grid size for time, x and y
        Nsamps = 1
        u1 = 0.2
        u2 = 0.2
        windmodel=WindSimple(u1,u2) # establish fixed wind model
        m = PDEModel(resolution=res,boundary=boundary,N_feat=N_feat,noiseSD=noiseSD,kernel=k,sensormodel=sensors,windmodel=windmodel,k_0=k_0,R=0)
        
        if non_neg:
            m.computeModelRegressors()
            Xnonneg = meshgridndim(m.boundary,10,True)
            nnc = NonNegConstraint(m,np.array([[]]),Xnonneg,thinning=5,burnin=100,jitter=0.02,verbose=True,meanZ = np.zeros(N_feat),covZ = np.eye(N_feat),startpointnormalised=True)
            Zs_nonneg = nnc.sample(Nsamps)
            z = Zs_nonneg[0,:]
        else:
            z = np.random.randn(N_feat)
        ####
        source=(m.computeSourceFromPhi(z))# Compute source
        conc=m.computeResponse(source) # Compute concentration - runs advection diffusion forward model
        Y= m.computeObservations(addNoise=True) # Compute observations with noise uses m.sensormodel for observation locations
        
        self.X = X
        self.Y = Y
        self.source = source
        self.conc = conc
        self.boundary = boundary
        self.m = m

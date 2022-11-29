import numpy as np
from advectionGP.sensors import FixedSensorModel 
from advectionGP.kernels import EQ 
from advectionGP.constraints import NonNegConstraint
from advectionGP.models.mesh_1d_ode_model import AdjointSecondOrderODEModel as ODEModel
from advectionGP.kernels import meshgridndim

class ODEModelSample():
    def __init__(self,ls=1,non_neg=False,N_feat=200,k_0=0.001,u=0.001,eta=0.001,Npoints=10):
        """
        Generates a sample from our ODE model (with different hyperparameters). Option available to restrict it to non-negative samples
        """
        tlocL = np.linspace(1,9,Npoints) 
        X= np.zeros((len(tlocL),2)) 

        X[:,0] = tlocL
        X[:,1] = X[:,0]+0.1

        sensors = FixedSensorModel(X,0.1) # establish sensor model arguments are sensor locations and spatial averaging
        
        boundary = ([0],[10])# edges of the grid - in units of time
        k = EQ(ls, 2.0) # generate EQ kernel arguments are lengthscale and variance
        res = [200] # grid size for time, x and y
        m = ODEModel(resolution=res,boundary=boundary,N_feat=N_feat,noiseSD=0.1,kernel=k,sensormodel=sensors,k_0=k_0,u=u,eta=eta)
        
        if non_neg:
            m.computeModelRegressors()
            Xnonneg = meshgridndim(m.boundary,200,True)
            nnc = NonNegConstraint(m,np.array([[]]),Xnonneg,thinning=2,burnin=20,jitter=0.002,verbose=True,meanZ = np.zeros(N_feat),covZ = np.eye(N_feat),startpointnormalised=True)
            Zs_nonneg = nnc.sample(1)
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

import numpy as np
from advectionGP.models.mesh_adr2d_model import AdjointAdvectionDiffusionReaction2DModel as PDEModel 
from advectionGP.sensors import FixedSensorModel #Builds sensor arrays to generate data for foward model or to generate observations for comparison
from advectionGP.kernels import EQ
from advectionGP.wind import WindSimple
from scipy.stats import norm
import matplotlib.pyplot as plt
class GaussianBump:
    def __init__(self,blobcentre = np.array([3,3,3]), blobsize = np.array([1,1,1]), blobheight = 20):
        """
        A single bump appears and disappears.
           
        Stores five variables:
        
            self.winddirdistributions = list of arrays [angle, percent]
            self.avgwinddirs = array of angles
            self.windspeeds = array of windspeeds
        """
        self.blobcentre = blobcentre
        self.blobsize = blobsize
        self.blobheight = blobheight
        k_0 = 0.01
        R=0
        noiseSD = 0.05
        N_feat=100
        boundary = ([0,0,0],[10,15,15])
        k = EQ(5, 2.0) #not used.
        res = [200,100,100]
        wind=np.cos(np.linspace(0,6*np.pi,res[1]))*0.5
        u=[]
        u.append(np.ones(res)*wind)
        u.append(np.ones(res)*0.0)
        windmodel=WindSimple(0.4,0.4)

        tlocL = np.linspace(1,8,5) # lower time
        xloc=np.linspace(2,11,6) # x locations
        yloc=np.linspace(2,11,6) # y locations
        sensN = len(xloc)*len(yloc) # total number of sensors 
        obsN = len(tlocL) # total time points at which an observation is taken
        X= np.zeros((obsN*sensN,4))
        X[:,0] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[0] #lower time
        X[:,2] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[1] # x location
        X[:,3] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[2] # ylocation
        X[:,1] = X[:,0]+1 # upper time

        sensors = FixedSensorModel(X,1)

        self.m = PDEModel(resolution=res,boundary=boundary,N_feat=N_feat,noiseSD=noiseSD,kernel=k,sensormodel=sensors,windmodel=windmodel,k_0=k_0,R=R)
        #z=np.random.normal(0,1.0,N_feat) # Generate z to compute source
        #sourceGT=(m.computeSourceFromPhi(z))
        source = np.zeros([50,30,30])

        source = self.getrealsource(self.m.coords)
        conc = self.m.computeResponse(source) # Compute concentration - runs advection diffusion forward model
        Y = self.m.computeObservations() # Compute observations with noise uses m.sensormodel for observation locations
        
        self.X = X
        self.Y = Y
        self.conc = conc
        self.source = source
        self.boundary = boundary
        
    def getrealsource(self,p):
        return self.blobheight*norm(0,1).pdf(np.linalg.norm((np.transpose(np.array(self.m.coords[:,...]),[1,2,3,0])-self.blobcentre)/self.blobsize,axis=3))        
        
    def plot(self):
        plt.figure(figsize=[5,15])    
        i=0
        for tstep in range(0,self.m.resolution[0],50):
            i+=1
            plt.subplot(8,2,i)
            plt.imshow(self.source[tstep,:,:])
            plt.clim([0,30])
            i+=1            
            plt.subplot(8,2,i)
            plt.imshow(self.conc[tstep,:,:])            
            plt.clim([0,5])

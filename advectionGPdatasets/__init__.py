import numpy as np

def proposeboundary(X):
    """
    Proposes a boundary ([tmin,xmin,ymin],[tmax,xmax,ymax]) given an X
    """
    minX,maxX = np.min(X[:,[0,2,3]],0),np.max(X[:,[1,2,3]],0)
    edge = (maxX-minX)*0.1 #10% edge
    boundary=(list(np.floor(minX-edge)),list(np.ceil(maxX+edge)))
    return boundary


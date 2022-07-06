import numpy as np    
import xlrd
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
from os.path import exists

class RoundHillExperiment:
    def __init__(self,df,meta,metdf,winddirdist,windspeed,obsX,obsY):
        """
        A single experiment
        """
        self.df = df
        self.meta = meta
        self.metdf = metdf
        self.winddirdist = winddirdist
        self.avgwinddir = np.sum(np.prod(winddirdist,axis=1))/100
        self.windspeed = windspeed
        self.X = np.array(obsX)
        self.Y = np.array(obsY)
        ang = np.deg2rad(self.avgwinddir)
        self.windX = np.cos(ang) * windspeed
        self.windY = np.sin(ang) * windspeed

        
class RoundHill:
    def __init__(self):
        """
        Analysis the RoundHill dataset.
        From https://www.harmo.org/jsirwin/RoundHillDiscussion.html,
        
        Link to raw data: 
        using RoundHill2: https://www.harmo.org/jsirwin/RHILL2Update01.XLS 
           (also https://www.harmo.org/jsirwin/RHILL1Update01.XLS)
           
        Stores five variables:
        
            self.dfs = list of dataframes (one for each experiment)
            self.meta = list of meta-data dictionaries (one for each experiment)
            self.metdf = dataframe of meteorological data
            self.winddirdistributions = list of arrays [angle, percent]
            self.avgwinddirs = array of angles
            self.windspeeds = array of windspeeds
            

        """
        fn = "RHILL2Update01.XLS"
        if not exists(fn):
            print("Downloading %s" % fn)
            urllib.request.urlretrieve("https://www.harmo.org/jsirwin/RHILL2Update01.XLS", fn)    
        workbook = xlrd.open_workbook(fn)
        sheet = workbook.sheet_by_name('1) SO2 Concentrations')

        headers = ['Post Number','?','Azimuth (Degrees)','WindDir','50m10min','100m10min','200m10min','Post Number A','50m3min','100m3min','200m3min','Post Number B','50m0.5min','100m0.5min','200m0.5min','','','Post Number C','Height (m)','vert50m10min','vert100m10min','vert200m10min']

        metdf = pd.read_excel('RHILL2Update01.XLS','3) Meteorology',skiprows=18)


        tablestarts = [i for i,r in enumerate(sheet.get_rows()) if 'Number' in str(r[0].value)]
        tableends = [i for i,r in enumerate(sheet.get_rows()) if 'Sum(WD)' in str(r[2].value)]
        self.experiments = []
        pl = {'10':3,'3':2,'0.5':1}
        
        for tbstart,tbend in zip(tablestarts,tableends):
            m = {}
            m['date'] = "%d %s %d" % ((sheet.cell(tbstart-3,0).value),sheet.cell(tbstart-3,1).value,sheet.cell(tbstart-3,2).value)
            m['run'] = str(sheet.cell(tbstart-3,3).value)
            
            df = pd.read_excel('RHILL2Update01.XLS','1) SO2 Concentrations',skiprows=tbstart,nrows=tbend-tbstart-1,names=headers)
        
 
            run = int(m['run'][4:])
            metdata = metdf[metdf.iloc[:,1]==run]
            windspeed = metdata['   U(m/s)'].to_numpy()[0]
            winddist = []

            obsX = []
            obsY = []
            for i,r in df.iterrows():
                try:
                    angle = float(r['Azimuth (Degrees)'])
                except ValueError:
                    continue

                for d in [50,100,200]:        

                    for t in ['10','3','0.5']:
                        if (r['Post Number']%2==1) and (t=='10'): continue #only even posts at time 10.
                        v = r['%dm%smin' % (d,t)]
                        if np.isnan(v): v=0

                        x = np.cos(np.deg2rad(angle))
                        y = np.sin(np.deg2rad(angle))
                        obsX.append([0,float(t)*60,d*x,d*y])
                        obsY.append(v)
                windpercent = r['WindDir']
                if i%2==1:
                    if np.isnan(windpercent): windpercent = 0
                    winddist.append([angle,windpercent])
            winddist = np.array(winddist)
            exp = RoundHillExperiment(df,m,metdata,winddist,windspeed,obsX,obsY)
            self.experiments.append(exp)

            assert np.abs(np.sum(winddist[:,1])-100)<1 #within 1%

        #self.avgwinddirs = np.array(self.avgwinddirs)            
        #self.windspeeds = np.array(self.windspeeds)
        

                
    def plot(self,plotwind=False):
        pl = {'10':3,'3':2,'0.5':1}
        for df,m in zip(self.dfs,self.meta):
            run = int(m['run'][4:])
            metdata = self.metdf[self.metdf.iloc[:,1]==run]
            windspeed = metdata['   U(m/s)']
            #winddir = metdata['  Sa(deg)']#THIS IS THE STD DEV OF THE WIND DIR...
            data = {}
            
            
            for i,r in df.iterrows():
                try:
                    angle = float(r['Azimuth (Degrees)'])
                except ValueError:
                    continue
                data[angle] = {}
                for d in [50,100,200]:        
                    data[angle][d] = {}
                    for t in ['10','3','0.5']:
                        if (r['Post Number']%2==1) and (t=='10'): continue #only even posts at time 10.
                        #print('%dm%smin' % (d,t))
                        v = r['%dm%smin' % (d,t)]
                        if np.isnan(v): v=0
                        
                        data[angle][d][float(t)] = v
                        x = np.cos(np.deg2rad(angle))
                        y = np.sin(np.deg2rad(angle))
                        plt.subplot(1,3,pl[t])
                        plt.plot(d*x,d*y,'ok',mfc='white',markersize=v/30)
                        plt.plot(d*x,d*y,'.',markersize=1)
                        plt.axis('equal')
                if plotwind:                        
                    windpercent = r['WindDir']
                    if not np.isnan(windpercent):
                        plt.plot([0,windpercent*windspeed*x],[0,windpercent*windspeed*y],'b-')
                
        for i,v in pl.items():
            plt.subplot(1,3,v)
            plt.title("%s minutes" % i)
            radwinddir = np.deg2rad(winddir)
            plt.plot(0,0,'bx')
    
    def info(self):
        return

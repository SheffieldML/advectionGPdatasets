import numpy as np    
import xlrd
import pandas as pd
import urllib.request
   
class RoundHill:
    def __init__(self):
    """
    Analysis the RoundHill dataset.
    From https://www.harmo.org/jsirwin/RoundHillDiscussion.html,
    
    Link to raw data: 
    using RoundHill2: https://www.harmo.org/jsirwin/RHILL2Update01.XLS 
       (also https://www.harmo.org/jsirwin/RHILL1Update01.XLS)
       
    Stores three variables:
        self.dfs = list of dataframes (one for each experiment)
        self.meta = list of meta-data dictionaries (one for each experiment)
        self.metdf = dataframe of meteorological data
    """
        
        urllib.request.urlretrieve("https://www.harmo.org/jsirwin/RHILL2Update01.XLS", "RHILL2Update01.XLS")    
        workbook = xlrd.open_workbook('RHILL2Update01.XLS')
        sheet = workbook.sheet_by_name('1) SO2 Concentrations')

        headers = ['Post Number','?','Azimuth (Degrees)','WindDir','50m10min','100m10min','200m10min','Post Number A','50m3min','100m3min','200m3min','Post Number B','50m0.5min','100m0.5min','200m0.5min','','','Post Number C','Height (m)','vert50m10min','vert100m10min','vert200m10min']

        tablestarts = [i for i,r in enumerate(sheet.get_rows()) if 'Number' in str(r[0].value)]
        tableends = [i for i,r in enumerate(sheet.get_rows()) if 'Sum(WD)' in str(r[2].value)]
        self.dfs = []
        self.meta = []
        for tbstart,tbend in zip(tablestarts,tableends):
            #print(tbstart)
            metadata = {}
            metadata['date'] = "%d %s %d" % ((sheet.cell(tbstart-3,0).value),sheet.cell(tbstart-3,1).value,sheet.cell(tbstart-3,2).value)
            metadata['run'] = str(sheet.cell(tbstart-3,3).value)
            self.meta.append(metadata)
            self.dfs.append(pd.read_excel('RHILL2Update01.XLS','1) SO2 Concentrations',skiprows=tbstart,nrows=tbend-tbstart-1,names=headers))
        
        
        self.metdf = pd.read_excel('RHILL2Update01.XLS','3) Meteorology',skiprows=18)
        
    
    def plot(self):
        for df,m in zip(self.dfs,self.meta):
            run = int(m['run'][4:])
            metdata = self.metdf[self.metdf.iloc[:,1]==run]
            windspeed = metdata['   U(m/s)']
            #winddir = metdata['  Sa(deg)']#THIS IS THE STD DEV OF THE WIND DIR...
            data = {}
            pl = {'10':3,'3':2,'0.5':1}
            plt.figure(figsize=[15,4])
            for i,r in df.iterrows():
                try:
                    angle = float(r['Azimuth (Degrees)'])
                except ValueError:
                    continue
                data[angle] = {}
                for d in [50,100,200]:        
                    data[angle][d] = {}
                    for t in ['10','3','0.5']:
                        if (r['Post Number']%2==1) and (t=='10'): continue
                        #print('%dm%smin' % (d,t))
                        v = r['%dm%smin' % (d,t)]
                        if np.isnan(v): v=0
                        #print(angle,d,v);
                        data[angle][d][float(t)] = v
                        x = np.cos(np.deg2rad(angle))
                        y = np.sin(np.deg2rad(angle))
                        plt.subplot(1,3,pl[t])
                        plt.plot(d*x,d*y,'ok',mfc='white',markersize=v/30)
                        plt.plot(d*x,d*y,'.',markersize=1)
                        plt.axis('equal')
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

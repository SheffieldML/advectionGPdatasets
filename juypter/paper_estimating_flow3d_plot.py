        
#Generate figure
rhms = []
plt.figure()
for rep in np.arange(1):
    fn = getfn(walls,s2,ls,k0,Nfeat,Npart,rep)
    rhm = pickle.load(open(fn,'rb'))
    rhms.append(rhm)
    for sample_at_peak,col,style,lab in zip([True,False],['b','g'],['-','--'],['around peak','whole domain']):
        sourcemasses = []        
        if sample_at_peak:
            sources = rhm.results['sources']['all'][0,:,(73-15):(73+15),(12-12):(12+20),:3].copy()
        else:
            sources = rhm.results['sources']['all'][0,:,:,:,:].copy()
        pickle.dump(sources,open('temp_'+lab+'.pkl','wb'))
        #sources[sources<0]=0
        #np.sum(sources)*np.prod(rhm.mInfer.getGridStepSize()[0][1:])
        mass = np.sum(sources,(1,2,3))*np.prod(rhm.mInfer.getGridStepSize()[0][1:])

        sourcemasses.append(mass)
        #plt.plot(times,mass[:-1])
        times = np.arange(rhm.mInfer.boundary[0][0],rhm.mInfer.boundary[1][0],rhm.mInfer.getGridStepSize()[0][0])[:-1]
        m = np.mean(sourcemasses,0)[:-1]/1e3 #grams from mg.
        s = (np.std(sourcemasses,0)/np.sqrt(13))[:-1]/1e3 #13 samples (finding rough SE! argh)
        #s = np.sqrt((np.var(sourcemasses,0)/13+single_sample_var))[:-1]/1e3
        #s = np.sqrt(single_sample_var)[:-1]/1e3
        #sourcemasses = np.array(sourcemasses)
        plt.plot(times,m,style+col,label=lab)
        plt.fill_between(times,m-s*1.96,m+s*1.96,color=col,alpha=0.2)
        plt.ylabel('Mass flow rate / $g\;s^{-1}$')
        plt.xlabel('Time / $s$')
        #plt.ylim([-5,12.5])#12.500])
        plt.xlim(rhm.mInfer.boundary[0][0],rhm.mInfer.boundary[1][0]-4)
    
    #plt.title("%0.3f" % k0)
    #plt.hlines(39.5e3,0,300,'r')
plt.grid()
plt.hlines(0,-200,1000)
plt.legend()
#plt.ylim([-3,3])
fn = getfn(walls,s2,ls,k0,Nfeat,Npart,fig=True)
plt.savefig(fn)

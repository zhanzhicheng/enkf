__author__ = 'zhanzc'
#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
# def initial_param():
#     enseNum=30
#     #avg_s, avg_e, avg_i, avg_beta, avg_gamma, avg_mu = [0.07, 1E-4 , 8.54E-7 ,0.07,0.2,0.1]
#     #std_s, std_e, std_i, std_beta, std_gamma, std_mu = [0.006, 8.95E-6, 1.37E-7, 0.01, 0.007, 0.01]
#     initial_avg = [0.07, 1.0E-4, 8.54E-7, 0.04, 0.1, 0.1]
#     initial_std = [0.006, 1.0E-6, 1.37E-8, 0.01, 0.07, 0.01]
#     initial_ensem=[]
#     for i in range(len(initial_avg)):
#         s=np.random.normal(initial_avg[i],initial_std[i],enseNum)
#         initial_ensem.append(s)
#     np.savetxt(r"data_out/init_ensemble.txt",np.array(initial_ensem))
#     return np.array(initial_ensem)
def seirsingle(susp,expose,infect,beta,theta,gamma):
    N=19612368.0    #北京市人口
    ds=0-(beta*susp*infect)
    de=beta*susp*infect-theta*expose
    di=theta*expose-gamma*infect
    dr=gamma*infect
    return [susp+ds,expose+de,infect+di,beta,theta,gamma]


def enkf_forword_timestep(ensemble, time):
    ensembleNum=ensemble.shape[1]
    matensemble=np.mat(ensemble)
    #只是存发病率
    matenkfforword=np.mat(np.zeros((time,ensembleNum)))
    for i in range(ensembleNum):
        for j in range(time):
            parameters=np.mat(seirsingle(matensemble[0,i],matensemble[1,i],matensemble[2,i],matensemble[3,i],matensemble[4,i],matensemble[5,i])).reshape(6,1)
            matenkfforword[j, i]=parameters[2,1]

#所有集合同时预测下一个时刻
#data上随机加上误差
def createDataperturbation(obserror,ensembleNum):
    dataperturbation=obserror*np.random.randn(1,ensembleNum)
    return np.mat(dataperturbation)

def createEnsemble(param_avg,param_std,ensembleNum):
    ensemble=[]
    for i in range(len(param_avg)):
        s=np.random.normal(param_avg[i],param_std[i],ensembleNum)
        ensemble.append(s)
    return np.mat(ensemble)

def enkfanalysis(ensemble,currentObsData,obs_error):
    #obs_error=1.0E-7
    obserrormat=np.mat([obs_error])
    obsHmatrix=np.mat([0,0,1,0,0,0])
    #matobsvalue=np.mat([obsvalue])
    #mean variable
    #matensemble=np.mat(ensemble)
    #ensembleNum=len(ensemble[0,:])
    ensembleNum=ensemble.shape[1]

    #print ensembleNum
    meanvariable=ensemble.mean(axis=1)
    bh=np.mat(np.zeros((6,1)))
    hbh=np.mat(np.zeros((1,1)))
    for i in range(ensembleNum):
        hx_hx=(np.mat(obsHmatrix)*np.mat(ensemble[:,i])-np.mat(obsHmatrix)*meanvariable)
        bh=(np.mat(ensemble[:,i]).reshape((6,1))-meanvariable)*(hx_hx.T)+bh
        hbh=hx_hx*(hx_hx.T)+hbh
    #print type(ensemble)
    #print ensembleNum
    bh=bh/(ensembleNum-1)
    hbh=hbh/(ensembleNum-1)
    # kalman gain
    kalmanGain=bh*((hbh+obserrormat).I)
    #print kalmanGain
    analysisensemble=np.mat(np.zeros((6,ensembleNum)))
    for i in range(ensembleNum):
        analysisensemble[:,i]=ensemble[:,i]+kalmanGain*(currentObsData[0,i]-obsHmatrix*ensemble[:,i])
        #print currentObsData[0,i]-obsHmatrix*ensemble[:,i]

    return analysisensemble
def enkfanalysis2(ensemble,currentObsData,obs_error):
    ensembleparam=ensemble[0:6,:]
    #print ensembleparam
    Ensize=ensembleparam.shape[1]
    simSize=ensembleparam.shape[0]
    MeaSize=currentObsData.shape[0]
    observationensemble=ensembleparam[2,:]
    A=ensembleparam
    Amean=np.tile(A.mean(axis=1),Ensize)
    dA=A-Amean
    dD=currentObsData-observationensemble

    MeasAvg=np.tile(observationensemble.mean(axis=1),Ensize)
    S=observationensemble-MeasAvg
    #添加观测误差
    COV=(1./float(Ensize-1))*np.dot(S,S.transpose())+math.sqrt(obs_error)
    #COV=(1./float(Ensize-1))*np.dot(S,S.transpose())

    B=np.linalg.solve(COV,dD)
    dAS=(1./float(Ensize-1))*np.dot(dA,S.transpose())
    Analysis=A+np.dot(dAS,B)
    #Analysis[4:6,:]=ensemble[4:6,:]
    return Analysis
def enkfforword(ensemble):
    ensembleNum=ensemble.shape[1]
    matensemble=np.mat(ensemble)
    matenkfforword=np.mat(np.zeros((6,ensembleNum)))
    for i in range(ensembleNum):
        matenkfforword[:,i]=np.mat(seirsingle(matensemble[0,i],matensemble[1,i],matensemble[2,i],matensemble[3,i],matensemble[4,i],matensemble[5,i])).reshape(6,1)
    return matenkfforword
#forecast time future
def forecastforword(start_time,datafile):
    data=np.loadtxt(datafile)
    timeseries=len(data[:,0])
    time=timeseries-start_time
    ensemblefile=r'data_out/analysis%d.dat' % start_time
    ensemble=np.loadtxt(ensemblefile)
    ensembleNum=ensemble.shape[1]
    forecastresult=np.mat(np.zeros((time,ensembleNum)))
    for i in range(ensemble.shape[1]):
        a = ensemble[:, i]
        for j in range(time):
            a=seirsingle(a[0],a[1],a[2],a[3],a[4],a[5])
            #print type(c)
            forecastresult[j,i]=a[2]
    forecastfile=r'data_out/forecast%d.dat'%start_time
    np.savetxt(forecastfile,forecastresult)
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], 'b*')
    for i in range(ensembleNum):
        plt.plot(range(start_time, timeseries), forecastresult[:, i])
    plt.show()
def main():
    year=2008
    ensembleNum=200
    obs_error=1E-7
    param_avg = [0.06, 0.0004, 1.11111E-05, 0.6, 0.07, 0.15]
    param_std = [1E-6, 1E-5, 1E-6, 0.006, 0.007, 0.004]
    ensemble = createEnsemble(param_avg, param_std, ensembleNum)
    #read data
    observationdata=r"data/SEIRdata_%d_10week.dat" % year
    data=np.loadtxt(observationdata)
    timeseries=len(data[:,1])
    obsdata=np.mat(np.loadtxt(observationdata)[:,1])

    #save data
    enkf=np.mat(np.zeros((np.shape(obsdata)[1],6)))
    analysis=np.mat(np.zeros((np.shape(obsdata)[1],6)))


    enkf[0,:]=np.mat(param_avg)
    analysis[0,:]=np.mat(param_avg)
    np.savetxt(r'data_out/ensemble1.dat',ensemble)
    np.savetxt(r'data_out/analysis1.dat',analysis)
    for i in range(np.shape(obsdata)[1]-1):
        dataperturbation=createDataperturbation(obs_error,ensembleNum)
        current_obsdata = np.mat(np.tile(obsdata[0, i + 1],ensembleNum))+dataperturbation
        matenkforword=enkfforword(ensemble)

        matanalysis=enkfanalysis2(matenkforword,current_obsdata,1E-29)
        analysis[i + 1,:] = matanalysis.mean(axis=1).reshape(1, 6)
        enkf[i + 1, :] = matenkforword.mean(axis=1).reshape(1, 6)
        ensemble=matanalysis
        analysisfilename = r'data_out/analysis%d.dat' % (i + 2)
        ensemblefilename = r'data_out/ensemble%d.dat' % (i + 2)
        np.savetxt(analysisfilename,matanalysis)
        np.savetxt(ensemblefilename,matenkforword)
    #save txt
    np.savetxt(r'data_out/analysis.dat',analysis)
    np.savetxt(r'data_out/ensemble.dat',enkf)
        #ensemble=matenkforword
    #plot1 assimilation result
    for i in range(6):
        plt.plot(enkf[:,i],'r-')
        plt.plot(analysis[:,i],'g-')
        if(i==2):
            plt.plot(data[:,1], 'bo')
        plt.show()
    plt.plot(enkf[:,5]/enkf[:,3],'r-')
    plt.show()
    #extrend
    #plot2 forecast result
    for i in range(1,39,1):
        from_time=i
        forecastforword(from_time,observationdata)
if __name__=="__main__":
    print 'start...'
    main()
    print 'finish'










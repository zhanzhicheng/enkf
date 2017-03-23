import numpy as np
import matplotlib.pyplot as plt

def seirstep(susp,expose,infect,beta,theta,gamma):
    N=19612368.0
    ds=0-(beta*susp*infect/N)
    de=beta*susp*infect/N-theta*expose
    di=theta*expose-gamma*infect
    dr=gamma*infect
    return [susp+ds,expose+de,infect+di,beta,theta,gamma]
def seirtime(steps,parameters):
    susp,expose,infect,beta,theta,gamma=[parameters[0],parameters[1], parameters[2],parameters[3], parameters[4], parameters[5]]
    variables=[]
    for i in range(steps):
        tempvar=seirstep(susp,expose,infect,beta,theta,gamma)
        susp, expose, infect, beta, theta, gamma = tempvar
        variables.append(tempvar)
    resultarry=np.array(variables)
    # for i in range(6):
    #     plt.plot(resultarry[:,i])
    #     plt.show()
# return a set of parameters
def multiseir(enseNum,initial_avg,initial_std):
    #enseNum=30
    #avg_s, avg_e, avg_i, avg_beta, avg_gamma, avg_mu = [0.07, 1E-4 , 8.54E-7 ,0.07,0.2,0.1]
    #std_s, std_e, std_i, std_beta, std_gamma, std_mu = [0.006, 8.95E-6, 1.37E-7, 0.01, 0.007, 0.01]
    initial_ensem=[]
    for i in range(len(initial_avg)):
        s=np.random.normal(initial_avg[i],initial_std[i],enseNum)
        initial_ensem.append(s)
    paramense=np.array(initial_ensem)
    return paramense
def main():
    initial_avg = [0.07, 1.0E-4, 8.54E-7, 0.04, 0.1, 0.1]
    initial_std = [0.006, 1.0E-6, 1.37E-7, 0.01, 0.07, 0.01]
    ensembleNum=50
    parameterset=multiseir(50,initial_avg,initial_std)# an ensemble of parameters
    allensemble=[]
    for i in range(ensembleNum):
        singleresult=seirtime(parameterset[:,i])
        allensemble.append(singleresult)
if __name__=="__main__":
    main();






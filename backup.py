__author__ = 'zhanzc'
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def initial_param():
    enseNum = 30
    # avg_s, avg_e, avg_i, avg_beta, avg_gamma, avg_mu = [0.07, 1E-4 , 8.54E-7 ,0.07,0.2,0.1]
    # std_s, std_e, std_i, std_beta, std_gamma, std_mu = [0.006, 8.95E-6, 1.37E-7, 0.01, 0.007, 0.01]
    initial_avg = [0.07, 1.0E-4, 8.54E-7, 0.04, 0.1, 0.1]
    initial_std = [0.006, 1.0E-6, 1.37E-7, 0.01, 0.07, 0.01]
    initial_ensem = []
    for i in range(len(initial_avg)):
        s = np.random.normal(initial_avg[i], initial_std[i], enseNum)
        initial_ensem.append(s)
    np.savetxt(r"data_out/init_ensemble.txt", np.array(initial_ensem))
    return np.array(initial_ensem)


def seirsingle(susp, expose, infect, beta, theta, gamma):
    N = 19612368.0  # 北京市人口
    ds = 0 - (beta * susp * infect / N)
    de = beta * susp * infect / N - theta * expose
    di = theta * expose - gamma * infect
    dr = gamma * infect
    return [susp + ds, expose + de, infect + di, beta, theta, gamma]


# 所有集合同时预测下一个时刻
def seirsimulti(paramsEnsem):
    # result=[]
    nextparam = np.zeros((len(paramsEnsem), len(paramsEnsem[0])))
    for i in range(len(paramsEnsem[0])):
        singleResult = seirsingle(paramsEnsem[0][i], paramsEnsem[1][i], paramsEnsem[2][i], paramsEnsem[3][i],
                                  paramsEnsem[4][i], paramsEnsem[5][i])
        for j in range(6):
            nextparam[j][i] = singleResult[j]
    return nextparam


def enkfanalysis(ensemble, obsvalue):
    obs_error = 1.0E-7
    obserrormat = np.mat([obs_error])
    obsHmatrix = np.mat([0, 0, 1, 0, 0, 0])
    matobsvalue = np.mat([obsvalue])
    # mean variable
    matensemble = np.mat(ensemble)
    ensembleNum = len(ensemble[0])
    meanvariable = np.mat(
        [np.mean(ensemble[0, :]), np.mean(ensemble[1, :]), np.mean(ensemble[2, :]), np.mean(ensemble[3, :]),
         np.mean(ensemble[4, :]), np.mean(ensemble[5, :])]).reshape((6, 1))
    bh = np.mat((6, 1))
    hbh = np.mat((1, 1))
    for i in range(ensembleNum):
        hx_hx = (np.mat(obsHmatrix) * np.mat(ensemble[:, i]) - np.mat(obsHmatrix) * meanvariable)
        bh = (np.mat(ensemble[:, i]).reshape((6, 1)) - meanvariable) * (hx_hx.T) + bh  # 6*1
        hbh = hx_hx * (hx_hx.T) + hbh  # 1*1
    bh = bh / (ensembleNum - 1)
    hbh = hbh / (ensembleNum - 1)
    # kalman gain
    kalmanGain = ph * ((hbh + obserrormat).I)
    analysisensemble = np.mat(np.zeors((6, ensembleNum)))
    for i in range(ensembleNum):
        analysisensemble[:, i] = matensemble[:, i] + kalmanGain * (matobsvalue - obsHmatrix * matensemble[:, i])
    return analysisensemble


def main():


def enkfprocess(paramPrevious, number):
    # 定义观测误差
    obs_error = 1.0E-7
    # 先验预测：
    newensmeble = seirsimulti(paramPrevious)
    # 定义观测矩阵(观测值只有一个)：
    obsHmatrix = np.array([0, 0, 1, 0, 0, 0])
    # 状态变量
    # stateVariable=np.array([])

    # 平均值数组
    meanArray = np.zeros((6, 1))
    for i in range(6):
        meanArray[i][0] = np.mean(newensmeble[i, :])
    ens_erro = np.ones((len(newensmeble), len(newensmeble[0])))
    for j in range(len(newensmeble)):
        for i in range(len(newensmeble[0])):
            ens_erro[j][i] = newensmeble[j][i] - meanArray[j]
    p_h = np.zeros((6, 1))
    h_p_h = np.array([0])
    for ii in range(len(ens_erro[0])):
        p_h = p_h + np.dot(np.array(ens_erro[:, ii]).reshape((6, 1)), (
        np.dot(Hmatrix, np.array(newensmeble[:, ii]).reshape((6, 1))) - np.dot(Hmatrix, meanArray)).T).reshape(6, 1)
        p1 = np.dot(Hmatrix, np.array(newensmeble[:, ii].reshape((6, 1)))) - np.dot(Hmatrix, meanArray)
        # print p1
        # print "-----"
        h_p_h = h_p_h + np.dot(p1, p1.T)
        # print h_p_h
    # print p_h

    p_h = p_h / (len(newensmeble[0]) * 1.0 - 1)
    h_p_h = h_p_h / (len(newensmeble[0]) * 1.0 - 1)
    # 计算增益矩阵
    k = np.dot(p_h, 1 / (h_p_h[0] + obs_error))
    # print k
    # 计算分析变量
    # 读取观测值：
    observationData = np.loadtxt(r"data/SEIRdata_09.dat")
    # 发病数据
    obs = observationData[number + 1, 1]
    result = np.zeros((len(newensmeble), len(newensmeble[0])))
    for i in range(len(result[0])):
        a = np.array(newensmeble[:, i]).reshape((6, 1)) + np.dot(k, obs - np.dot(Hmatrix,
                                                                                 np.array(newensmeble[:, i]).reshape(
                                                                                     (6, 1)))).reshape((6, 1))
        for j in range(len(result)):
            result[j][i] = a[j]
    result_out = []
    for i in range(6):
        result_out.append(np.mean(result[i, :]))
    with open("data_out/result.txt", "a") as f:
        f.write("%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
        result_out[0], result_out[1], result_out[2], result_out[3], result_out[4], result_out[5]))
    return result


def plot():
    data = np.loadtxt(r"data_out/result.txt")
    obsfile = np.loadtxt(r"data/SEIRdata_09.dat")
    for i in range(6):
        plt.plot(data[:, i])
        # plt.plot(obsfile[:,1])
        plt.show()


def simulateeach():
    for i in range(len(current_ensemble[0])):
        timeseries = 52
        result = []
        a, b, c, d, e, f = [current_ensemble[0][i], current_ensemble[1][i], current_ensemble[2][i],
                            current_ensemble[3][i], current_ensemble[4][i], current_ensemble[5][i]]
        for j in range(timeseries):
            nextstate = seirsingle(a, b, c, d, e, f)
            a, b, c, d, e, f = [nextstate[0], nextstate[1], nextstate[2], nextstate[3], nextstate[4], nextstate[5]]
            result.append(nextstate)
        fileName = "data_out/seir_%.d.txt" % (i)
        result = np.array(result)
        np.savetxt(fileName, result)


if __name__ == "__main__":
    f = open(r"data_out/result.txt", "w")
    f.truncate()
    current_ensemble = initial_param()
    f.write("%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n" % (
    current_ensemble[0, 0], current_ensemble[1, 0], current_ensemble[2, 0], current_ensemble[3, 0]
    , current_ensemble[4, 0], current_ensemble[5, 0]))
    f.close()
    for i in range(51):
        current_ensemble = enkfprocess(current_ensemble, i)
        print "finish"
        print i
    plot()
    # simulateeach()
    # data=np.loadtxt(r"data_out/seir_2.txt");
    # plt.plot(data[:,2])
    # plt.show()






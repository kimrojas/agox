########################################################################################
#
# This program is designed to make it easy to make proper statistics for
# evolutionary algorithms. It can be used in two differnet ways:
#
# 1: Call the program from the terminal, followed be the files you want statistics for.
#    If you have two data sets you want to compre it could look like this:
#    $ python survival_stats.py event_file1.npy event_file2.npy
#    If you only want statistics for one 
#    The results will be saved in a map called stats/ so you can make all
#    the displayed yourself. Each file have a format of (x,y,y_LCB,y_UCB,censorings),
#    Hazard only holds (x,y)
#    A last input can be given to set labels. it shold be of this from:
#    labels=[label1,label2,label3]
#
# 2: Import the function survival_stats() from this script and give it two lists
#    of inputs. The first should be a list of all the times when an event or censoring 
#    occured. The second second should be a list of binaries indicating if an event 
#    occured at the corresponding time. That is 0 for censorings and 1 for events. 
#    List of list can also be used for input, and will additionally result in log-rank
#    tests being made between the inputs
#
# Event files should be .npy files holding a 2 x n array. eg. [[5,4,10][1,1,0]].
# the first vector holds the times for the EA runs. either the time when the best
# structure was found or when the run ended. The second vector should hold a list
# of zero and ones, where a 1 indicate that the best structure was found, and a 0
# indicate that it was not.
#
########################################################################################
import os
import sys
import numpy as np
import scipy.stats as st
from scipy.special import erfinv
from copy import copy
from agox.utils.kernelregression import KernelRegression # This should point to your kernel ridge modul
from matplotlib.pyplot import *

# Define a class to be used KRR
class Comp(object):
    def get_features(self,x):
        return x
    def get_similarity(self,f1,f2):
        return abs(f1-f2)

# Function for the log-rank test
def logrank(n1,d1,t1,n2,d2,t2):
    """This function returns the p-value for a log-rank test
    
    Inputs:
    n1: Number at risk in population 1 at times indicated by t1
    d1: Number of events in population 1 at times indicated by t1
    t1: times used with the two above inputs
    n2: Number at risk in population 2 at times indicated by t2
    d2: Number of events in population 2 at times indicated by t2
    t2: times used with the two above inputs

    output:
    p-value

    """

    # The first part here is just collecting and ordering the inputs
    # for the calculations
    n1 = copy(n1)
    d1 = copy(d1)
    t1 = copy(t1)
    n2 = copy(n2)
    d2 = copy(d2)
    t2 = copy(t2)
    n = []
    n_1 = []
    n_2 = []
    d = []
    d_1 = []
    d_2 = []
    while t1 or t2:
        if t1 and t2:
            if t1[0] < t2[0]:
                n_1.append(n1.pop(0))
                n_2.append(n_2[-1])
                n.append(n_1[-1]+n_2[-1])
                d_1.append(d1.pop(0))
                d_2.append(0)
                d.append(d_1[-1]+d_2[-1])
                t1.pop(0)
            elif t1[0] > t2[0]:
                n_1.append(n_1[-1])
                n_2.append(n2.pop(0))
                n.append(n_1[-1]+n_2[-1])
                d_1.append(0)
                d_2.append(d2.pop(0))
                d.append(d_1[-1]+d_2[-1])
                t2.pop(0)
            elif t1[0] == t2[0]:
                n_1.append(n1.pop(0))
                n_2.append(n2.pop(0))
                n.append(n_1[-1]+n_2[-1])
                d_1.append(d1.pop(0))
                d_2.append(d2.pop(0))
                d.append(d_1[-1]+d_2[-1])
                t1.pop(0)
                t2.pop(0)
        elif t1:
            n_1.append(n1.pop(0))
            n_2.append(n_2[-1])
            n.append(n_1[-1]+n_2[-1])
            d_1.append(d1.pop(0))
            d_2.append(0)
            d.append(d_1[-1]+d_2[-1])
            t1.pop(0)
        elif t2:
            n_1.append(n_1[-1])
            n_2.append(n2.pop(0))
            n.append(n_1[-1]+n_2[-1])
            d_1.append(0)
            d_2.append(d2.pop(0))
            d.append(d_1[-1]+d_2[-1])
            t2.pop(0)
    # This is where the actual test is performed
    e_1 = []
    v = []
    for i in range(len(n)):
        e1 = n_1[i]*d[i]/float(n[i])
        e_1.append(e1)
        v1 = (d[i] * n_1[i]/float(n[i]) * (1-n_1[i]/float(n[i])) * (n[i]-d[i])) / float(n[i]-1)
        v.append(v1)
    Z = np.sum(np.array(d_1)-np.array(e_1)) / np.sqrt(np.sum(v))
    return st.norm.sf(abs(Z))*2

# This is the real function of interest
def survival_stats(times,events,alpha=0.95,sigma=5000,show_plot=True,save=True,get_hazard=True,labels=[]):
    """This function calculateds a number of statistics that may beof interest

    inputs:
    times:     Either a single list of times or a list of list of times
    events:    Either a single list of events or a list of list of events.
               0 indicates a censoring and 1 indicates an event.
    alpha:     Is the size of the confidence bound given for the functions.
               Default is 0.95
    sigma:     Is used for the kernel size for the kernel smoothing used to creat
               the hazard curve. Lower the number if the curve seems to flat and
               raise it if the crve is to spikey. Default is 5000
    show_plot: Default is True. Change to False if you don't want to see
               the plots
    save:      Default is True. Change to False if you don't want the statistics saved

    Output: (KM,CDF,NA,Hazard,censoring,logrank_res)
    KM:          Kaplan-Meier. List of tuples containing: time, value of KM , LCB of KM,
                 UCB of KM.
    CDF:         Cumultative distribution function. List of tuples containing: time,
                 value of CDF , LCB of CDF, UCB of CDF.
    NA:          Nelson-Aalen. List of tuples containing: time, value of NA , LCB of NA, UCB of NA.
    Hazard:      List of tuples containing: time, value of Hazard
    censoring:   List of list indicating if censorings occured at the times given by the KM times
    logrank_res: The results of the log-rank tests arranged in a matrix

    All the outer lists are used to seperate multiple inputs.
    """
    # Arrange the input into a standard format
    if hasattr((times[0]),'__len__'):
        n_inputs = len(times)
    else:
        n_inputs = 1
        times = [times]
        events = [events]
    # calculate a z value from the given alpha
    z = np.sqrt(2)*erfinv(2*(alpha+(1-alpha)/2.)-1)

    # Change the input to conviniant format
    time = []
    censoring = []
    n = []
    d = []
    for i in range(n_inputs):
        time.append([0])
        censoring.append([False])
        n.append([len(times[i])])
        d.append([0])
        ds = 0 # dead or censord at this timestep
        sort_index = np.argsort(times[i])
        for j in sort_index:
            if times[i][j] == time[i][-1]:
                ds += 1
                if events[i][j]:
                    d[i][-1] += 1
                else:
                    censoring[i][-1] = True
            else:
                time[i].append(times[i][j])
                n[i].append(n[i][-1]-ds)
                ds = 1
                if events[i][j]:
                    d[i].append(1)
                    censoring[i].append(False)
                else:
                    d[i].append(0)
                    censoring[i].append(True)
        censoring[i][-1] = False
        censoring[i] = np.array(censoring[i])

    # Make Kaplan-Meier 
    KM = []
    for i in range(n_inputs):
        S = [1]
        for j in range(1,len(time[i])):
            S.append(S[-1]*(n[i][j]-d[i][j])/float(n[i][j]))
        KM.append((np.array(time[i]),np.array(S)))

    # Make confidence bounds for Kaplan-Meier
    KM_CB = []
    for i in range(n_inputs):
        S_LCB = [1]
        S_UCB = [1]
        temp = 0
        for j in range(1,len(time[i])):
            if KM[i][1][j] == 1:
                c_L = 1
                c_U = 1
            elif n[i][j] != d[i][j]:
                temp += d[i][j]/float(n[i][j]*(n[i][j]-d[i][j]))
                V = temp/(float(np.log(KM[i][1][j])**2) + 0.000001)
                c_L = np.log(-np.log(KM[i][1][j])) + z*np.sqrt(V)
                c_U = np.log(-np.log(KM[i][1][j])) - z*np.sqrt(V)
            else:
                V = temp/(float(np.log(KM[i][1][j-1])**2) + 0.000001)
                c_L = np.log(-np.log(KM[i][1][j-1])) + z*np.sqrt(V)
                c_U = np.log(-np.log(KM[i][1][j-1])) - z*np.sqrt(V)
            S_LCB.append(np.exp(-np.exp(c_L)))
            S_UCB.append(np.exp(-np.exp(c_U)))
        KM_CB.append((np.array(time[i]),np.array(S_LCB),np.array(S_UCB)))

    # Gather all KM stuff
    for i in range(n_inputs):
        KM[i] = (KM[i][0],KM[i][1],KM_CB[i][1],KM_CB[i][2])

    # Make Cumultative distribution function
    CDF = []
    CDF_CB = []
    for i in range(n_inputs):
        CDF.append((KM[i][0],1-KM[i][1]))
        CDF_CB.append((KM_CB[i][0],1-KM_CB[i][1],1-KM_CB[i][2]))

    # Gather all CDF stuff 
    for i in range(n_inputs):
        CDF[i] =(CDF[i][0],CDF[i][1],CDF_CB[i][1],CDF_CB[i][2])

    # Make Nelson-Aalen
    NA = []
    for i in range(n_inputs):
        L = [0]
        for j in range(1,len(time[i])):
            L.append(L[-1]+(d[i][j]/float(n[i][j])))
        NA.append((np.array(time[i]),np.array(L)))

    # Make confidence bounds for Nelson-Aalen
    NA_CB = []
    for i in range(n_inputs):
        CH_LCB = [0]
        CH_UCB = [0]
        temp = 0
        for j in range(1,len(time[i])):
            if n[i][j] != 1:
                temp += (n[i][j]-d[i][j])*d[i][j]/float((n[i][j]-1)*n[i][j]**2)
            if temp != 0:
                CH_LCB.append(NA[i][1][j]*np.exp(-z*np.sqrt(temp)/float(NA[i][1][j])))
                CH_UCB.append(NA[i][1][j]*np.exp(z*np.sqrt(temp)/float(NA[i][1][j])))
            else:
                CH_LCB.append(0)
                CH_UCB.append(0)
        NA_CB.append((np.array(time[i]),np.array(CH_LCB),np.array(CH_UCB)))

    # Gather all NA stuff 
    for i in range(n_inputs):
        NA[i] =(NA[i][0],NA[i][1],NA_CB[i][1],NA_CB[i][2])

    # Make Hazard function
    if get_hazard:
        comp = Comp()
        Hazard = []
        #Hazard_CB = []
        for i in range(n_inputs):
            t = np.arange(NA[i][0][-1])
            fm = NA[i][0]
            sm = np.zeros([len(fm),len(fm)])
            for j in range(len(fm)):
                for k in range(j):
                    sm[j,k] = comp.get_similarity(fm[j],fm[k])
            sm = sm + sm.T
            kreg = KernelRegression(data_values=NA[i][1], feature_matrix=fm, similarity_matrix=sm, comp=comp)
            kreg.sigma = sigma
            CH_S, _ = kreg.predict_values(t)
            H = CH_S[1:]-CH_S[:-1]
            t = t[:-1]+0.5
            Hazard.append((t,H))


    # Here the log-rank tests are made
    logrank_res = np.ones([n_inputs]*2)

    if show_plot: # if show_plot=False don't print stuff
        print('\nLog-rank test:')
    for i in range(n_inputs):
        if show_plot:
            print('')
        for j in range(i+1,n_inputs):
            logrank_res[i,j] = logrank(n[i],d[i],time[i],n[j],d[j],time[j]) # function call
            logrank_res[j,i] = logrank_res[i,j]
            if show_plot:
                print('({:2},{:2}) p-value: {:.3}'.format(i,j,logrank_res[i,j]))
    if show_plot:
        print('')

    if show_plot:# make plots
        labels += range(len(labels),n_inputs)
        f, axarr = subplots(2,2,sharex='col',figsize=[16,9])
        colors = ['b','r','g','y','c','m']
        max_time = 0
        for i in range(n_inputs):
            try:
                axarr[0,0].fill_between(KM[i][0], KM[i][2], KM[i][3], step='post', facecolor=colors[i%len(colors)], alpha=0.1)
            except:
                axarr[0,0].fill_between(KM[i][0], KM[i][2], KM[i][3], facecolor=colors[i%len(colors)], alpha=0.1)
            axarr[0,0].step(KM[i][0], KM[i][1],where='post',c=colors[i%len(colors)],label=labels[i])
            axarr[0,0].plot(KM[i][0][censoring[i]],KM[i][1][censoring[i]],marker='+',c='k')
            if KM[i][0][-1] > max_time:
                max_time = KM[i][0][-1]
        axarr[0,0].set_ylabel('Survival Rate')
        axarr[0,0].set_xlim([0,max_time])
        axarr[0,0].set_ylim([0,1])
        axarr[0,0].set_yticks(np.linspace(0,1,6))
        axarr[0,0].set_yticklabels(['{} %'.format(int(i)) for i in np.linspace(0,100,6)])
        axarr[0,0].set_title('Survival')
        axarr[0,0].legend(loc=3)

        for i in range(n_inputs):
            try:
                axarr[0,1].fill_between(CDF[i][0], CDF[i][2], CDF[i][3], step='post', facecolor=colors[i%len(colors)], alpha=0.1)
            except:
                axarr[0,1].fill_between(CDF[i][0], CDF[i][2], CDF[i][3], facecolor=colors[i%len(colors)], alpha=0.1)
            axarr[0,1].step(CDF[i][0], CDF[i][1],where='post',c=colors[i%len(colors)])
            axarr[0,0].plot(CDF[i][0][censoring[i]],CDF[i][1][censoring[i]],marker='+',c='k')
        axarr[0,1].set_ylabel('Succes Rate')
        axarr[0,1].set_xlim([0,max_time])
        axarr[0,1].set_ylim([0,1])
        axarr[0,1].set_yticks(np.linspace(0,1,6))
        axarr[0,1].set_yticklabels(['{} %'.format(int(i)) for i in np.linspace(0,100,6)])
        axarr[0,1].set_title('CDF')

        max_NA = 0
        for i in range(n_inputs):
            try:
                axarr[1,0].fill_between(NA[i][0], NA[i][2], NA[i][3], step='post', facecolor=colors[i%len(colors)], alpha=0.1)
            except:
                axarr[1,0].fill_between(NA[i][0], NA[i][2], NA[i][3], facecolor=colors[i%len(colors)], alpha=0.1)
            axarr[1,0].step(NA[i][0], NA[i][1],where='post',c=colors[i%len(colors)])
            axarr[0,0].plot(NA[i][0][censoring[i]],NA[i][1][censoring[i]],marker='+',c='k')
            if NA[i][3][-1] > max_NA:
                max_NA = NA[i][3][-1]
        axarr[1,0].set_xlabel('EA Attempts')
        axarr[1,0].set_ylabel('Cum Hazard')
        axarr[1,0].set_title('Cum Hazard')
        axarr[1,0].set_xlim([0,max_time])
        axarr[1,0].set_ylim([0,max_NA])

        if get_hazard:
            max_hazard = 0
            for i in range(n_inputs):
                axarr[1,1].plot(Hazard[i][0],Hazard[i][1],c=colors[i%len(colors)])
                if np.max(Hazard[i][1]) > max_hazard:
                    max_hazard = np.max(Hazard[i][1]) 
            axarr[1,1].set_title('Hazard')
            axarr[1,1].set_xlabel('EA Attempts')
            axarr[1,1].set_ylabel('Hazard Rate')
            axarr[1,1].set_xlim([0,max_time])
            axarr[1,1].set_ylim([0,max_hazard*1.25])

    if save: # Save files
        cwd = os.getcwd()
        if not os.path.exists(os.path.join('stats')):
            os.makedirs(os.path.join('stats'))
        for i in range(n_inputs):
            name = str(i)
            name_survival = name+'_Survival'
            if os.path.isfile(os.path.join(cwd,'stats',name_survival+'.npy')):
                n_name = 1
                while os.path.isfile(os.path.join(cwd,'stats',name_survival+'({})'.format(n_name)+'.npy')):
                    n_name += 1
                name_survival += '({})'.format(n_name)
            name_survival += '.npy'
            np.save(os.path.join(cwd,'stats',name_survival),(KM[i][0],KM[i][1],KM[i][2],KM[i][3],censoring[i]))

            name_CDF = name+'_CDF'
            if os.path.isfile(os.path.join(cwd,'stats',name_CDF+'.npy')):
                n_name = 1
                while os.path.isfile(os.path.join(cwd,'stats',name_CDF+'({})'.format(n_name)+'.npy')):
                    n_name += 1
                name_CDF += '({})'.format(n_name)
            name_CDF += '.npy'
            np.save(os.path.join(cwd,'stats',name_CDF),(CDF[i][0],CDF[i][1],CDF[i][2],CDF[i][3],censoring[i]))

            name_CH = name+'_CumHaz'
            if os.path.isfile(os.path.join(cwd,'stats',name_CH+'.npy')):
                n_name = 1
                while os.path.isfile(os.path.join(cwd,'stats',name_CH+'({})'.format(n_name)+'.npy')):
                    n_name += 1
                name_CH += '({})'.format(n_name)
            name_CH += '.npy'
            np.save(os.path.join(cwd,'stats',name_CH),(NA[i][0],NA[i][1],NA[i][2],NA[i][3],censoring[i]))

            if get_hazard:
                name_H = name+'_Haz'
                if os.path.isfile(os.path.join(cwd,'stats',name_H+'.npy')):
                    n_name = 1
                    while os.path.isfile(os.path.join(cwd,'stats',name_H+'({})'.format(n_name)+'.npy')):
                        n_name += 1
                    name_H += '({})'.format(n_name)
                name_H += '.npy'
                np.save(os.path.join(cwd,'stats',name_H),(Hazard[i][0],Hazard[i][1]))

    if show_plot:
        show()

    if get_hazard:
        return (KM,CDF,NA,Hazard,censoring,logrank_res)
    else:
        return (KM,CDF,NA,censoring,logrank_res)

if __name__ == '__main__':
    cwd = os.getcwd()
    # Check how many inputs are given
    try:
        n_inputs = len(sys.argv)-1
    except:
        print('At least one input must be given for this program.')
        raise
    try:
        if 'label' in sys.argv[-1]:
            n_inputs -= 1
            label_str = sys.argv[-1]
            index1 = label_str.find('[')+1
            label_str = label_str[index1:]
            labels = []
            while label_str.find(',') != -1:
                index2 = label_str.find(',')
                label = label_str[:index2]
                labels.append(label)
                label_str = label_str[index2+1:]
            index2 = label_str.find(']')
            label =label_str[:index2]
            labels.append(label)
        else:
            labels = []
    except:
        print('labels should be given as the last input with a format like:\n'\
              +'labels=[label1,label2,label3]')
        raise

    # Prepare the files for input into the function
    times = [None]*n_inputs
    events = [None]*n_inputs
    for i in range(n_inputs):
        times[i], events[i] = np.load(os.path.join(cwd,sys.argv[i+1]))
    # Run function
    survival_stats(times,events,labels=labels)

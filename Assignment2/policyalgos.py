import numpy as np
import pulp

def valueIteration(mdp_data):
    np.random.seed(1)
    sizeA = mdp_data['numActions']
    sizeS = mdp_data['numStates']
    flag = True if mdp_data['type']=='episodic' else False
    termS = mdp_data['termStates']
    Tr = mdp_data['tranProb']
    Re = mdp_data['rewards']
    gamma = mdp_data['gamma']
    Vp = np.zeros(sizeS)
    Pis = np.zeros(sizeS)
    Vs = np.random.random(sizeS)
    if flag:
        Vs[termS] = 0

    t = 0
    while(np.max(np.abs(Vs-Vp)) > 1e-9 or t==0):
        Vp = np.copy(Vs)
        Vs = np.max(np.sum(Tr*(Re + gamma*Vp), axis=2), axis=1)
        if flag:
            Vs[termS] = 0
        t+=1
    Pis = np.argmax(np.sum(Tr*(Re + gamma*Vs), axis=2), axis=1)
    return Vs, Pis

def adjustPolicy(Qpi, pi, S, A):
    newPi = np.zeros(S)
    for s in range(S):
        gtQval = [i for i in range(A) if Qpi[s, i] > Qpi[s, pi[s]]]
        if gtQval:
            selAct = np.random.choice(gtQval)
        else:
            selAct = pi[s]
        newPi[s] = selAct

    if np.sum(newPi==pi) == S:
        cond = False
    else:
        cond = True

    return newPi.astype('int'), cond

def howardPolicyIteration(mdp_data):
    np.random.seed(20)
    sizeA = mdp_data['numActions']
    sizeS = mdp_data['numStates']
    flag = True if mdp_data['type']=='episodic' else False
    termS = mdp_data['termStates']
    Tr = mdp_data['tranProb']
    Re = mdp_data['rewards']
    gamma = mdp_data['gamma']
    Vp = np.zeros(sizeS)
    Vs = np.random.random(sizeS)
    Pis = np.random.randint(sizeA, size=sizeS, dtype='int')
    allStates = np.arange(0, sizeS, dtype='int')
    if flag:
        Vs[termS] = 0
        Pis[termS] = 0
    t=0
    cond = True # True = policy is changed if it is not change we will stop
    while(cond or np.max(np.abs(Vs-Vp)) > 1e-9 or t==0):
        Vp = np.copy(Vs)
        Qpi = np.sum(Tr*(Re + gamma*Vp), axis=2)
        Pis, cond = adjustPolicy(Qpi, Pis, sizeS, sizeA)      
        Vs = Qpi[allStates, Pis]      
        if flag:
            Vs[termS] = 0
            Pis[termS] = 0
        t+=1
    return Vs, Pis
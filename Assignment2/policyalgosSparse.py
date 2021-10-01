import numpy as np
import pulp as pl
from pulp.apis.coin_api import PULP_CBC_CMD
from pulp.constants import LpMaximize

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
    Pis = np.zeros(sizeS, dtype='int')
    Vs = np.random.random(sizeS)

    if flag:
        Vs[termS] = 0

    t = 0

    while(np.max(np.abs(Vs-Vp)) > 1e-9 or t==0):
        Vp = np.copy(Vs)
        for s in range(sizeS):
            if flag and (s in termS):
                Vs[s] = 0
                continue

            overA = []
            for a in range(sizeA):
                sumOverS = 0
                for sd in range(sizeS):
                    sumOverS += (Tr.get((s, a, sd), 0))*(Re.get((s, a, sd), 0) + gamma*Vp[sd])
                overA.append(sumOverS)
            
            Pis[s] = np.argmax(overA)
            Vs[s] = np.max(overA)
        t+=1
    
    return Vs, Pis

def adjustPolicy(Qpi, pi, S, A, flag, termS):
    newPi = np.zeros(S)
    for s in range(S):
        if flag and (s in termS):
            continue
        gtQval = [i for i in range(A) if Qpi.get((s, i), 0) > Qpi.get((s, pi[s]), 0)]
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

    if flag:
        Vs[termS] = 0
        Pis[termS] = 0
    t=0
    cond = True # True = policy is changed if it is not change we will stop

    while(cond):
        Vp = np.copy(Vs)
        Qpi = {}
        
        for s in range(sizeS):
            if flag and (s in termS):
                continue

            for a in range(sizeA):
                sumOverS = 0
                for sd in range(sizeS):
                    sumOverS += (Tr.get((s, a, sd), 0))*(Re.get((s, a, sd), 0) + gamma*Vp[sd])
                Qpi[(s, a)] = sumOverS
        
        Pis, cond = adjustPolicy(Qpi, Pis, sizeS, sizeA, flag, termS) 
        for s in range(sizeS):
            if flag and (s in termS):
                Vs[s] = 0
                continue
            Vs[s] = Qpi.get((s, Pis[s]), 0)
        
        t2 = 0 
        while(np.max(np.abs(Vs-Vp)) > 1e-9 or t2==0):
            Vp = np.copy(Vs)
            for s in range(sizeS):
                if flag and (s in termS):
                    Vs[s] = 0
                    continue

                a = Pis[s]
                sumOverS = 0
                for sd in range(sizeS):
                    sumOverS += (Tr.get((s, a, sd), 0))*(Re.get((s, a, sd), 0) + gamma*Vp[sd])
                Vs[s] = sumOverS
            t2+=1
        t+=1
    
    return Vs, Pis

def linearProgramming(mdp_data):
    np.random.seed(30)
    sizeA = mdp_data['numActions']
    sizeS = mdp_data['numStates']
    flag = True if mdp_data['type']=='episodic' else False
    termS = mdp_data['termStates']
    Tr = mdp_data['tranProb']
    Re = mdp_data['rewards']
    gamma = mdp_data['gamma']
    
    problem = pl.LpProblem("MDP_Problem", LpMaximize)
    # declaring all the variables
    Vp = pl.LpVariable.dicts("Vs", range(0, sizeS), cat='Continuous')

    # objective function
    problem += -pl.lpSum([Vp[i] for i in range(0, sizeS)])

    # constraints
    for s in range(sizeS):
        for a in range(sizeA):
            problem += pl.lpSum([(Tr.get((s, a, i), 0))*(Re.get((s, a, i), 0) + gamma*Vp[i]) for i in range(sizeS)]) <= Vp[s]
    
    if flag:
        for i in termS:
            problem += Vp[i]==0
    
    problem.solve(PULP_CBC_CMD(msg=0))

    Vs = np.array([Vp[i].varValue for i in range(sizeS)])

    Pis = np.zeros(sizeS)
    for s in range(sizeS):
        if flag and (s in termS):
            Pis[s] = 0
            continue
        overA = []
        for a in range(sizeA):
            sumOverS = 0
            for sd in range(sizeS):
                sumOverS += (Tr.get((s, a, sd), 0))*(Re.get((s, a, sd), 0) + gamma*Vs[sd])
            overA.append(sumOverS)            
        Pis[s] = np.argmax(overA)
    
    return Vs, Pis
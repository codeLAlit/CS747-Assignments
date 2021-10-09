import numpy as np
import pulp as pl
from pulp.apis.coin_api import PULP_CBC_CMD
from pulp.constants import LpMaximize, LpMinimize

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
        Qs = np.zeros((sizeS, sizeA))
        for key in Tr.keys():
            Qs[key[0], key[1]] += Tr[key]*(Re[key]+gamma*Vp[key[2]])
        Pis = np.argmax(Qs, axis=1)
        Vs = np.max(Qs, axis=1)
        if flag:
            Vs[termS] = 0
        t+=1
    
    return Vs, Pis

def adjustPolicy(Qpi, pi, S, A, flag, termS):
    newPi = np.zeros(S)
    for s in range(S):
        if flag and (s in termS):
            continue
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

    if flag:
        Vs[termS] = 0
        Pis[termS] = 0
    t=0
    cond = True # True = policy is changed if it is not change we will stop

    while(cond):
        Vp = np.copy(Vs)
        Qpi = np.zeros((sizeS, sizeA))
        for key in Tr.keys():
            Qpi[key[0], key[1]] += Tr[key]*(Re[key]+gamma*Vp[key[2]])
        if flag:
            Qpi[termS, :] = 0
        
        Pis, cond = adjustPolicy(Qpi, Pis, sizeS, sizeA, flag, termS) 
        Vs = Qpi[np.arange(0, sizeS), Pis]
            
        # Solving system of linar equations to get V(s) for sparse and large states action is not
        # efficient. So the way around I thought of is it iteratively converge it to a solution.
        # At first glance the lower section may appear exactly similar to Value iteration,
        # but its not. Here I am convergining it using a fixed policy found using HPI.
        # Later I found the same thing in literature too, to back correctness of my implementation.
        # Refer https://www.ics.uci.edu/~dechter/publications/r42a-mdp_report.pdf same as point 4 in refernces
        t2 = 0
        while(np.max(np.abs(Vs-Vp)) > 1e-9 or t2==0):
            Vp = np.copy(Vs)
            Qs = np.zeros((sizeS, sizeA))
            for key in Tr.keys():
                Qs[key[0], key[1]] += Tr[key]*(Re[key]+gamma*Vp[key[2]])
            Vs = Qs[np.arange(0, sizeS), Pis]
            if flag:
                Vs[termS] = 0
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
    # some processing for faster execution
    ts = {}
    for key in Tr.keys():
        if ts.get(key[0]):
            if ts[key[0]].get(key[1]) :
                ts[key[0]][key[1]].append(key[2])
            else:
                ts[key[0]][key[1]] = [key[2]]
        else:
            ts[key[0]] = {}
            ts[key[0]][key[1]] = [key[2]]
    # problem definition
    problem = pl.LpProblem("MDP_Problem", LpMaximize)
    # declaring all the variables
    Vp = pl.LpVariable.dicts("Vs", range(0, sizeS), lowBound=0.0, cat='Continuous')

    # objective function
    problem += -pl.lpSum([Vp[i] for i in range(0, sizeS)])

    # constraints
    for s in range(sizeS):
        if flag and (s in termS):
            problem += Vp[s]==0
            continue
        if not ts.get(s):
            problem += Vp[s] >= 0
            continue
        for a in ts[s]:
            problem += pl.lpSum([Tr[(s, a, sd)]*(Re[(s, a, sd)] + gamma*Vp[sd]) for sd in ts[s][a]]) <= Vp[s]
        # for a in range(sizeA):
        #     problem += pl.lpSum([(Tr.get((s, a, i), 0))*(Re.get((s, a, i), 0) + gamma*Vp[i]) for i in sdall]) <= Vp[s]
    

    problem.solve(PULP_CBC_CMD(msg=0))

    Vs = np.array([Vp[i].varValue for i in range(sizeS)])

    Pis = np.zeros(sizeS, dtype='int')
    for s in range(sizeS):
        if flag and (s in termS):
            Pis[s] = 0
            continue
        if not ts.get(s):
            Vs[s] = 0
            continue
        maxOverA = 0
        act = 0
        for a in ts[s]:
            sumOverS = 0
            for sd in ts[s][a]:
                sumOverS += (Tr[(s, a, sd)]*(Re[(s, a, sd)] + gamma*Vs[sd]))
            if sumOverS > maxOverA:
                maxOverA = sumOverS
                act = a            
        Pis[s] = act
    
    return Vs, Pis
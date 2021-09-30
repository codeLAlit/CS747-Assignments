import numpy as np
import pulp

def valueIteration(mdp_data):
    np.random.seed(1)

    flag = True if mdp_data['type']=='episodic' else False
    termS = mdp_data['termStates']
    Tr = mdp_data['tranProb']
    Re = mdp_data['rewards']
    gamma = mdp_data['gamma']
    Vp = np.zeros(mdp_data['numStates'])
    Pis = np.zeros(mdp_data['numStates'])
    Vs = np.random.random(mdp_data['numStates'])
    if flag:
        Vs[termS] = 0

    t = 0
    while(np.max(np.abs(Vs-Vp)) > 1e-9 or t==0):
        Vp = np.copy(Vs)
        Vs = np.max(np.sum(Tr*(Re + gamma*Vp), axis=2), axis=1)
        if flag:
            Vs[termS] = 0
        t+=1
    Pis = np.argmax(np.sum(Tr*(Re + gamma*Vp), axis=2), axis=1)
    return Vs, Pis

def howardPolicyIteration(mdp_data):
    np.random.seed(1)

    flag = True if mdp_data['type']=='episodic' else False
    termS = mdp_data['termStates']
    Tr = mdp_data['tranProb']
    Re = mdp_data['rewards']
    gamma = mdp_data['gamma']
    Vp = np.zeros(mdp_data['numStates'])
    Pis = np.zeros(mdp_data['numStates'])
    Vs = np.random.random(mdp_data['numStates'])
    if flag:
        Vs[termS] = 0

    t = 0
    
    return Vs, Pis
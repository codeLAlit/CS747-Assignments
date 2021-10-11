import numpy as np
import argparse

from numpy.lib.shape_base import split

class MDPCreator():
    def __init__(self, policy_file, states_file):
        self.policy_file = policy_file
        self.states_file = states_file

        self.numStates = 0
        self.numActions = 9
        self.termS = []
        self.states_to_id = {}
        self.pstates = []

        self.parsePolicyFile()
        self.envStates = list(self.envPS.keys())
        self.parseStateFile()
        self.warmup()
        self.printMDPFile()

    def giveStateString(self, curr, act, val):
        l = []
        for i in range(self.numActions):
            if i==act:
                l.append(str(val))    
            else:
                l.append(curr[i])             
        return ''.join(l)

    def getStateType(self, state):
        # normal = 0, win state achieved = 1, draw = 2
        for i in range(3):
            if state[i*3+0]==state[i*3+1] and state[i*3+1]==state[i*3+2] and state[i*3+2]!=str(0):
                return 1
        for i in range(3):
            if state[i]==state[3+i] and state[3+i]==state[6+i] and state[6+i]!=str(0):
                return 1
        if state[0]==state[4] and state[4]==state[8] and state[8]!=str(0):
            return 1
        if state[2]==state[4] and state[4]==state[6] and state[6]!=str(0):
            return 1
        numZeros = 0
        for i in range(len(state)):
            if state[i]=='0':
                numZeros+=1
        if numZeros==0:
            return 2
        return 0

    def getNextState(self, pCurrState, act):
        '''
            Returns:
            (nextState, type, reward, probability)
            type = 'T': terminal, 'N': non terminal
        '''
        nextStates = []
        paction = self.giveStateString(pCurrState, act, self.pID)
        if paction in self.envStates:
            pis = self.envPS[paction]
            for i in range(len(pis)):
                if pis[i] == 0:
                    continue
                else:
                    envaction = self.giveStateString(paction, i, self.envID)
                    stateType = self.getStateType(envaction)
                    if stateType == 1:
                        # env lost
                        nextStates.append(('WIN', 'T', 1, pis[i]))
                    elif stateType == 2:
                        # draw combination
                        nextStates.append(('DRORL', 'T', 0, pis[i]))
                    elif stateType == 0:
                        # everthing is normal game continues
                        nextStates.append((envaction, 'N', 0, pis[i]))
        else:
            stateType = self.getStateType(paction)
            if stateType == 1:
                # player lost
                nextStates.append(('DRORL', 'T', 0, 1.0))
            elif stateType == 2:
                # draw combination
                nextStates.append(('DRORL', 'T', 0, 1.0))
        return nextStates

    def parseStateFile(self):
        with open(self.states_file, 'r') as f:
            data = f.readlines()
        self.pID = 1 if self.envID==2 else 2
        
        for line in data:
            splits = line.split()
            self.states_to_id[splits[0]] = self.numStates
            self.pstates.append(splits[0])
            self.numStates += 1
        
    def parsePolicyFile(self):
        with open(self.policy_file, 'r') as f:
            data = f.readlines()
        envPS = {}
        envID = int(data[0].split()[0])
        for line in data[1:]:
            splits = line.split()
            envPS[splits[0]] = [float(i) for i in splits[1:]]
        
        self.envID = envID
        self.envPS = envPS

    def warmup(self):
        # pstates = self.pstates
        # termS = set()
        # for s in pstates:
        #     for a in range(self.numActions):
        #         if s[a]!='0':
        #             continue
        #         nxtStates = self.getNextState(s, a)
        #         terminals = [i[0] for i in nxtStates if i[1]=='T']
        #         for t in terminals:
        #             termS.add(t)
        # for t in termS:
        #     self.states_to_id[t] = self.numStates
        #     self.termS.append(self.numStates)
        #     self.numStates += 1   
        self.states_to_id['WIN'] = self.numStates
        self.termS.append(self.numStates)
        self.numStates += 1
        self.states_to_id['DRORL'] = self.numStates
        self.termS.append(self.numStates)
        self.numStates += 1     

    def printMDPFile(self):
        print("numStates", self.numStates)
        print("numActions", self.numActions)
        print("end", ' '.join(map(str, self.termS)))
        for s in self.pstates:
            for a in range(self.numActions):
                if s[a]!='0':
                    print("transition", self.states_to_id[s], a, self.states_to_id['DRORL'], -1e10, 1)
                    continue
                nxtStates = self.getNextState(s, a)
                for sd in nxtStates:
                    # sd = (sd', type, reward, prob)
                    if sd[3]==0:
                        continue
                    print("transition", self.states_to_id[s], a, self.states_to_id[sd[0]], sd[2], sd[3])
        print("mdptype", "episodic")
        print("discount ", 1)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--states", type=str, required=True)
    args = parser.parse_args()

    policy_file = args.policy
    states_file = args.states

    MDPCreator(policy_file, states_file)
    
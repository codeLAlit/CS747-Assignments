import argparse

class ValueDecoder():
    def __init__(self, value_policy_file, state_file, pID):
        self.value_file = value_policy_file
        self.state_file = state_file
        self.pID = pID
        self.numActions = 9
        self.numStates = 0
        self.pstates = []
        self.policy = []
        self.value = []
        self.probs = []

        self.parseStates()
        self.parseValue()
        self.getProbs()

        self.printPolicy()

    def parseStates(self):
        with open(self.state_file, 'r') as f:
            data = f.readlines()
        for i in data:
            self.pstates.append(i.split()[0])
        self.numStates = len(self.pstates)

    def parseValue(self):
        with open(self.value_file, 'r') as f:
            data = f.readlines()        
        data = [i.split() for i in data if i!='\n']

        for i in data:
            self.value.append(float(i[0]))
            self.policy.append(float(i[1]))
    
    def getProbs(self):
        for i in range(self.numStates):
            prob = [0.0]*self.numActions
            prob[int(self.policy[i])] = 1.0
            self.probs.append(prob)

    def printPolicy(self):
        print(self.pID)
        for i in range(self.numStates):
        #     lineStr = []
        #     lineStr.append(self.pstates[i])
        #     for j in self.probs[i]:
        #         lineStr.append(str(j))
            print(self.pstates[i], ' '.join(map(str, self.probs[i])))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-policy", type=str, required=True)
    parser.add_argument("--states", type=str, required=True)
    parser.add_argument("--player-id", type=str, required=True)
    args = parser.parse_args()

    value_policy_file = args.value_policy
    state_file = args.states
    pID = args.player_id

    ValueDecoder(value_policy_file, state_file, pID)
import numpy as np
import pandas as pd

data = pd.DataFrame(data=pd.read_csv('Training.csv'))

concepts = np.array(data.iloc[:,0:-1])
print(concepts)
target = np.array(data.iloc[:,-1])
print(target)

print("initialization of specific_h and general_h")
specificHypo = concepts[0].copy()    
#print(specificHypo)
generalHypo = [["?" for i in range(len(specificHypo))] for i in range(len(specificHypo))]
#print(generalHypo)

# The learning iterations
for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(len(specificHypo)):
            if h[x] != specificHypo[x]:
                specificHypo[x] = '?'
                generalHypo[x][x] = '?'

    if target[i] == "No":
        for x in range(len(specificHypo)):
            # For negative hyposthesis change values only  in G
            if h[x] != specificHypo[x]:
                generalHypo[x][x] = specificHypo[x]
            else:
                generalHypo[x][x] = '?'
    print("Steps of Candidate Elimination Algorithm",i+1)
    print("specific Hypo ",i+1)
    print(specificHypo)
    print("general Hypo ", i+1)
    print(generalHypo,"\n")

# find indices where we have empty rows, meaning those that are unchanged
indices = [i for i, val in enumerate(generalHypo) if val == ['?', '?', '?', '?', '?', '?']]
for i in indices:
    # remove those rows from generalHypo
    generalHypo.remove(['?', '?', '?', '?', '?', '?'])

print("Final specificHypo:", specificHypo, sep="\n")
print("Final generalHypo:", generalHypo, sep="\n")

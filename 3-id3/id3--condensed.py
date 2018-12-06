import pandas as pd 
tennisDF = pd.DataFrame.from_csv('tennis-ise.csv')
print("\n Given Play Tennis Data Set:\n\n",tennisDF)

tennisDF.keys()[0]
print(tennisDF.keys())
#Function to calculate the entropy of probaility of observations
# -p*log2*p
    
def entropyOfList(a_list):  
    #a_list --> yes/no col
    from collections import Counter
    cnt = Counter(x for x in a_list)
    probOfYesNo = [x / 14 for x in cnt.values()]  # x means no of YES/NO num_instances = len(a_list)*1.0   # = 14
    print("\n Classes:",min(cnt),max(cnt)) #min(cnt)--> Yes/no
    print(" \n Probabilities of Class {0} is {1}:".format(min(cnt),min(probOfYesNo)))
    print(" \n Probabilities of Class {0} is {1}:".format(max(cnt),max(probOfYesNo)))

    import math
    return sum( [-prob*math.log(prob, 2) for prob in probOfYesNo] ) #summation(-Plog2P)

print("\n  INPUT DATA SET FOR ENTROPY CALCULATION:\n", tennisDF['PlayTennis'])
print("\n Total Entropy of PlayTennis Data Set:",entropyOfList(tennisDF['PlayTennis']))

def information_gain(df, splitAttr, targetAttr):
    print("Information Gain Calculation of ",splitAttr,":")
    df_split = df.groupby(splitAttr)
    for name,group in df_split:
            print("Name:\n",name)
            print("Group:\n",group)
    # Calculate Entropy for Target Attribute, as well as
    # Proportion of Obs in Each Data-Split
    nobs = len(df.index) * 1.0
    print("NOBS",nobs)
    #Function to use for aggregating the data. 1st and only param : dict of column names -> functions (or list of functions).
    #returns a DataFrame object 
    aggDF = df_split.agg({targetAttr : [entropyOfList, lambda x: len(x)/nobs] })[targetAttr]
    aggDF.columns = ['Entropy', 'PropObservations']
    print("DFAGGENT",aggDF)

    # Calculate Information Gain:
    new_entropy = sum( aggDF['Entropy'] * aggDF['PropObservations'] )
    old_entropy = entropyOfList(df[targetAttr])
    return old_entropy - new_entropy

print('Info-gain for Outlook is :'+str( information_gain(tennisDF, 'Outlook', 'PlayTennis')))
print('Info-gain for Humidity is: ' + str( information_gain(tennisDF, 'Humidity', 'PlayTennis')))
print('Info-gain for Wind is:' + str( information_gain(tennisDF, 'Wind', 'PlayTennis')))
print('Info-gain for Temperature is:' + str( information_gain(tennisDF, 'Temperature','PlayTennis')))

def id3(df, targetAttr, attribute_names, default_class=None):
    from collections import Counter
    cnt = Counter(x for x in df[targetAttr])# class of YES /NO
    if len(cnt) == 1:
        return next(iter(cnt))  # next input data set, or raises StopIteration when EOF is hit.
    elif df.empty or (not attribute_names):
        return default_class  
    else:
        default_class = max(cnt.keys()) #No of YES and NO Class
        informationGain = [information_gain(df, attr, targetAttr) for attr in attribute_names] #
        index_of_max = informationGain.index(max(informationGain)) # Index of Best Attribute
        # Choose Best Attribute to split on:
        best_attr = attribute_names[index_of_max]
        
        # Create an empty tree, to be populated in a moment
        tree = {best_attr:{}} # Iniiate the tree with best attribute as a node 
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
        # Split dataset
        # On each split, recursively call this algorithm.
        # populate the empty tree with subtrees, which
        # are the result of the recursive call
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                        targetAttr,
                        remaining_attribute_names,
                        default_class)
            tree[best_attr][attr_val] = subtree
        return tree
# Get Predictor Names (all but 'class')
attribute_names = list(tennisDF.columns)
print("List of Attributes:", attribute_names) 
attribute_names.remove('PlayTennis') #Remove the class attribute 
print("Predicting Attributes:", attribute_names)
# Run Algorithm:
from pprint import pprint
tree = id3(tennisDF,'PlayTennis',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
#print(tree)
pprint(tree)
attribute = next(iter(tree))
print("Best Attribute :\n",attribute)
print("Tree Keys:\n",tree[attribute].keys())

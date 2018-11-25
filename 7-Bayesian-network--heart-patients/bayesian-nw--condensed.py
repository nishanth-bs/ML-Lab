import bayespy.nodes as bpnode
from bayespy.nodes import Dirichlet, Categorical, MultiMixture
import numpy as np
import csv 
from colorama import init
from colorama import Fore, Back, Style
init()

# Define Parameter Enum values
ageEnum = {'SuperSeniorCitizen':0, 'SeniorCitizen':1, 'MiddleAged':2, 'Youth':3, 'Teen':4}
genderEnum = {'Male':0, 'Female':1}
familyHistoryEnum = {'Yes':0, 'No':1}
dietEnum = {'High':0, 'Medium':1, 'Low':2} #Calory intake
lifeStyleEnum = {'Athlete':0, 'Active':1, 'Moderate':2, 'Sedetary':3}	
cholesterolEnum = {'High':0, 'BorderLine':1, 'Normal':2}
heartDiseaseEnum = {'Yes':0, 'No':1}

with open('heart_disease_data.csv') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    data = []
    for x in dataset:	
        data.append([ageEnum[x[0]],genderEnum[x[1]],familyHistoryEnum[x[2]],dietEnum[x[3]],lifeStyleEnum[x[4]],cholesterolEnum[x[5]],heartDiseaseEnum[x[6]]])
    """
    data-->[[0, 0, 0, 1, 3, 0, 0], [0, 1, 0, 1, 3, 0, 0], [1, 0, 1, 0, 2, 1, 0],
    [4, 0, 0, 1, 3, 2, 1],[3, 1, 0, 0, 0, 2, 1], [2, 0, 0, 1, 1, 0, 0], [4, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 3, 0, 0],[3, 1, 0, 0, 0, 2, 1], [1, 1, 1, 0, 0, 2, 0], [4, 1, 1, 1, 2, 0, 0]]
    """
data = np.array(data)
N = len(data)
print(N)

p_age = Dirichlet(1.0*np.ones(5))#used to classify text in a document to a particular topic.
age = Categorical(p_age, plates=(N,))#a sequence of unique values and no missing values
age.observe(data[:,0])

p_gender = Dirichlet(1.0*np.ones(2))
gender = Categorical(p_gender, plates=(N,))
gender.observe(data[:,1])

p_familyhistory = Dirichlet(1.0*np.ones(2))
familyhistory = Categorical(p_familyhistory, plates=(N,))
familyhistory.observe(data[:,2])

p_diet = Dirichlet(1.0*np.ones(3))
diet = Categorical(p_diet, plates=(N,))
diet.observe(data[:,3])

p_lifestyle = Dirichlet(1.0*np.ones(4))
lifestyle = Categorical(p_lifestyle, plates=(N,))
lifestyle.observe(data[:,4])

p_cholesterol = Dirichlet(1.0*np.ones(3))
cholesterol = Categorical(p_cholesterol, plates=(N,))
cholesterol.observe(data[:,5])

# Prepare nodes and establish edges
# np.ones(2) ->  HeartDisease has 2 options Yes/No
# plates(5, 2, 2, 3, 4, 3)  ->  corresponds to options present for domain values 
p_heartdisease = Dirichlet(np.ones(2), plates=(5, 2, 2, 3, 4, 3))
heartdisease = MultiMixture([age, gender, familyhistory, diet, lifestyle, cholesterol], Categorical, p_heartdisease)
heartdisease.observe(data[:,6])
p_heartdisease.update()
m = 0
while m == 0:
    print("\n")
    res = MultiMixture([int(input('Enter Age: ' + str(ageEnum))),
                        int(input('Enter Gender: ' + str(genderEnum))),
                        int(input('Enter FamilyHistory: ' + str(familyHistoryEnum))),
                        int(input('Enter dietEnum: ' + str(dietEnum))),
                        int(input('Enter LifeStyle: ' + str(lifeStyleEnum))),
                        int(input('Enter Cholesterol: ' + str(cholesterolEnum)))],
                       Categorical, p_heartdisease)
    res = res.get_moments()[0][heartDiseaseEnum['Yes']]
    print("Probability(HeartDisease) = " +  str(res))
    m = int(input("Enter for Continue:0, Exit :1  "))

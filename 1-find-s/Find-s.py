import csv
hypo=['%','%','%','%','%','%']

with open('Training_examples.csv') as csv_file:
    readcsv = csv.reader(csv_file, delimiter=',')
    print(readcsv)
    data = []
    print("\nThe given training examples are:") 
    for row in readcsv:     
        #print(row)
        if row[-1].upper() == "YES":
            data.append(row)
            
print("\nThe positive examples are:")
for i in data:
    print(i)
print("The steps of the Find-s algorithm are\n",hypo);

hypo = data[0][:-1]
print(hypo)
for i in range(len(data)):    
    for k in range(len(hypo)):
        if hypo[k].upper()!=data[i][k].upper():
            hypo[k]='?';
    print(hypo);
print("\nThe maximally specific Find-s hypothesis for the given training examples is");
print(hypo);



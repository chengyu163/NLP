import sys


dev_labels=open(sys.argv[1],'r').read()   #Reads in the dev_labels file
predicted_labels=open(sys.argv[2],'r').read() #Reads in the predicted labels file

true_positive =0
false_positive=0
for (i,j) in zip(dev_labels.split('\n'), predicted_labels.split('\n')):
    if ':' not in j:
        i = i.split(':')[0]
    if i == j:
         true_positive += 1
    else:
        false_positive += 1
        
print(true_positive)
print(false_positive)
print(true_positive/(false_positive+true_positive))
import sys
from sklearn.metrics import accuracy_score

dev_labels=open(sys.argv[1],'r').read()   #Reads in the dev_labels file
predicted_labels=open(sys.argv[2],'r').read() #Reads in the predicted labels file
preds=predicted_labels.splitlines()
dev_labels_list=dev_labels.splitlines()
if ':' not in preds[0]:
    dev_labels_list = [i.split(':')[0] for i in dev_labels_list]
print(accuracy_score(dev_labels_list, preds))
        


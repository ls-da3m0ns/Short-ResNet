import os,inspect
import tensorflow as tf
import numpy as np

PICK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def test(neuralnet, dataset, batch_size):
    try: neuralnet.load_params()
    except: print("Param loading failed")
    
    print("Test .. ")
    
    confusion_mat = np.zeros((dataset.num_classes,dataset.num_classes),np.int32)
    
    while True:
        images,labels,terminator = dataset.next_test_batch(1)
        loss,accuracy,class_score = neuralnet.step(x=images,y=labels,train=False)
        
        label,logit = np.argmax(labels[0]),np.argmax(class_score)
        confusion_mat[label,logit] += 1
        
        if(terminator): break
        
    print("Confusion Matrix ")
    print(confusion_mat)
        
    tot_precision,tot_recall,tot_f1score = 0,0,0
    digonal = 0
        
    for index in range(dataset.num_classes):
        precision = confusion_mat[index,index] / np.sum(confusion_mat[:,index])
        recall = confusion_mat[index,index] / np.sum(confusion_mat[index,:])
        f1score = 2*(precision * recall)/(precision + recall)
            
        tot_precision += precision
        tot_recall += recall
        tot_f1score += f1score
            
        digonal += confusion_mat[index,index]
        print("Class-%d | Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
            %(index, precision, recall, f1score))
        
    accuracy = digonal / np.sum(confusion_mat)
    print("accuracy %.5f, precision %.5f, | recall %.5f, | f1score %.5f"%(accuracy,tot_precision/dataset.num_classes,tot_recall/dataset.num_classes,tot_f1score/dataset.num_classes))

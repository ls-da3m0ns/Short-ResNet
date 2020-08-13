import os,inspect
import tensorflow as tf
import numpy as np

PICK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(neuralnet,dataset,epochs,batch_size,normalize=True):
    print("Training to %d epochs (%d of minibatch size)"%(epochs,batch_size))
    iteration = 0
    test_sq = 20
    test_size = test_sq**2
    
    for epoch in range(epochs):
        while True:
            images,labels,terminator = dataset.next_train_batch(batch_size)
            loss,accuracy,class_score = neuralnet.step(x=images,y=labels, iteration = iteration, train=True)
            iteration += 1
            if terminator : break
            neuralnet.save_params()
        print("Epoch [%d / %d] (%d  iteration) Loss:%.5f, Acc:%.5f"%(epoch,epochs,iteration,loss,accuracy))


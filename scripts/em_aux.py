### LIBRARIES
import numpy as np
import operator

def compute_dice(prediction,ground_truth):
    intersection = np.logical_and(prediction, ground_truth)
    dice_score   = (2 * np.sum(intersection)) / (np.sum(prediction) + np.sum(ground_truth))
    return dice_score

def dice_metric(prediction,ground_truth,labels=1):
    dice_scores = []
    for L in range(int(labels)):
        dice_scores.append(compute_dice((prediction==(L+1))*1,(ground_truth==(L+1))*1))
    dice_scores = np.asarray(dice_scores) 
    print("Dice Score (Overall): " + str(np.mean(dice_scores)))
    return dice_scores

def restructure_KMeans(K,FV):
    restruc_KMpredict       = np.ones(len(K.predict(FV)))
    restruc_KMcentroids     = np.zeros((K.cluster_centers_).shape)
    min_index, min_value    = min(enumerate((K.cluster_centers_)[:,0]), key=operator.itemgetter(1))
    max_index, max_value    = max(enumerate((K.cluster_centers_)[:,0]), key=operator.itemgetter(1))  
    restruc_KMcentroids[0]  = (K.cluster_centers_)[min_index]
    restruc_KMcentroids[2]  = (K.cluster_centers_)[max_index]
    
    if (min_index+max_index==1):
       restruc_KMcentroids[1]=(K.cluster_centers_)[2]
    elif (min_index+max_index==2):
       restruc_KMcentroids[1]=(K.cluster_centers_)[1]
    elif (min_index+max_index==3):
       restruc_KMcentroids[1]=(K.cluster_centers_)[0]
    
    restruc_KMpredict[(K.predict(FV))==min_index] = 0
    restruc_KMpredict[(K.predict(FV))==max_index] = 2
    
    restruc_KMpredict[restruc_KMpredict==2] = 3
    restruc_KMpredict[restruc_KMpredict==1] = 2
    restruc_KMpredict[restruc_KMpredict==3] = 1
    
    return restruc_KMpredict+1, restruc_KMcentroids

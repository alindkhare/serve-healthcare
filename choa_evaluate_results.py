import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,precision_recall_curve,auc,accuracy_score
import os

def ReadLines(file, n=3000):
    list_files = []
    i = 0
    with open(file,'r') as f:
        for line in f:
            line = line.strip()
            list_files.append(line)
            i += 1
            if i > n:
                break
    return list_files

pred_dir = 'choa_pred_results/'

true_labels = np.loadtxt(pred_dir + 'labels.txt')
model_list = ReadLines(pred_dir + 'model_list.txt')

def evaluate_ensemble_models_per_patient(b, debug=False):
    num_per_day = 24*60*2 # 30 seconds
    
    roc_aucs = []
    pr_aucs = []
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    if debug:
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                    print(model_list[i]) 
                    
    for n in range(len(true_labels)):
        # only evaluate label 1
        if true_labels[n] == 0:
            continue

        # get ensemble scores
        y_scores = []
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                yhat = np.loadtxt('{0}{1}/{1}_{2}.{3}.out'.format(pred_dir, n, model_list[i], int(true_labels[n])))
                y_scores.append(yhat)
        
        y_scores = np.nan_to_num(np.array(y_scores))
                
        if len(y_scores) > 1:
            y_scores = np.mean(y_scores,0)
        else:
            y_scores = y_scores[0]
        
        y_scores = np.concatenate([y_scores[:num_per_day * 2], y_scores[-num_per_day:]])
        y_true = np.concatenate([np.zeros(num_per_day*2), np.ones(num_per_day)])
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        y_pred = (y_scores > 0.5)+ 0

        roc_aucs.append(roc_auc_score(y_true, y_scores))
        pr_aucs.append(pr_auc)
        f1_scores.append(f1_score(y_true, y_pred))
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true,y_pred))
        recalls.append(recall_score(y_true,y_pred))
            
  
    return np.mean(roc_aucs), np.std(roc_aucs), np.mean(pr_aucs), np.std(pr_aucs), np.mean(f1_scores), np.std(f1_scores), np.mean(precisions), np.std(precisions), np.mean(recalls), np.std(recalls), np.mean(accuracies), np.std(accuracies)

def evaluate_ensemble_models_with_history_per_patient(b, obs_w_30sec=1, debug=False):
    num_per_day = 24*60*2 # 30 seconds
    
    roc_aucs = []
    pr_aucs = []
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    if debug:
        print('Observation window {0} x 30 seconds'.format(obs_w_30sec))
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                    print(model_list[i]) 
                    
    for n in range(len(true_labels)):
        # only evaluate label 1
        if true_labels[n] == 0:
            continue

        # get ensemble scores
        y_scores = []
        for i in range(len(model_list)):
            if i == len(b):
                break
            if b[i] == 1:
                yhat = np.loadtxt('{0}{1}/{1}_{2}.{3}.out'.format(pred_dir, n, model_list[i], int(true_labels[n])))
                y_scores.append(yhat)
        
        y_scores = np.nan_to_num(np.array(y_scores))
                
        if len(y_scores) > 1:
            y_scores = np.mean(y_scores,0)
        else:
            y_scores = y_scores[0]
        
        if obs_w_30sec == 1:
            y_agg_scores = np.concatenate([y_scores[:(num_per_day*2)], y_scores[-num_per_day:]])
            y_true = np.concatenate([np.zeros(num_per_day*2), np.ones(num_per_day)])
        else:
            y_agg_scores = []
            
            tmp_scores = y_scores[:(num_per_day*2)]
            for s in range(0,num_per_day*2-obs_w_30sec):
                y_agg_scores.append(np.mean(tmp_scores[s:(s+obs_w_30sec)])) 
                
            tmp_scores = y_scores[-num_per_day:]
            for s in range(0,num_per_day-obs_w_30sec):
                y_agg_scores.append(np.mean(tmp_scores[s:(s+obs_w_30sec)]))
                
            y_true = np.concatenate([np.zeros(num_per_day*2-obs_w_30sec), np.ones(num_per_day-obs_w_30sec)])
        
        
        y_scores = np.array(y_agg_scores)
        y_pred = (y_scores > 0.5)+ 0
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        roc_aucs.append(roc_auc_score(y_true, y_scores))
        pr_aucs.append(pr_auc)
        f1_scores.append(f1_score(y_true, y_pred))
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true,y_pred))
        recalls.append(recall_score(y_true,y_pred))
            
  
    return np.mean(roc_aucs), np.std(roc_aucs), np.mean(pr_aucs), np.std(pr_aucs), np.mean(f1_scores), np.std(f1_scores), np.mean(precisions), np.std(precisions), np.mean(recalls), np.std(recalls), np.mean(accuracies), np.std(accuracies)

if __name__ == "__main__":

    roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1,f1_std,precision,precision_std,recall,recall_std,accuracy, accuracy_std = evaluate_ensemble_models_per_patient(b = [1,1], debug=True)

    print(roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1,f1_std,precision,precision_std,recall,recall_std,accuracy, accuracy_std)

    roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1,f1_std,precision,precision_std,recall,recall_std,accuracy, accuracy_std = evaluate_ensemble_models_with_history_per_patient(b=[1,1], obs_w_30sec=1, debug=True)

    print(roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1,f1_std,precision,precision_std,recall,recall_std,accuracy, accuracy_std)

    roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1,f1_std,precision,precision_std,recall,recall_std,accuracy, accuracy_std = evaluate_ensemble_models_with_history_per_patient(b=[1,1], obs_w_30sec=10, debug=True)

    print(roc_auc,roc_auc_std,pr_auc,pr_auc_std,f1,f1_std,precision,precision_std,recall,recall_std,accuracy, accuracy_std)
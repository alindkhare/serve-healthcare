import numpy as np
from sklearn.metrics import roc_auc_score

pred_dir = '/shared/choa/KDD_2020/pred_results/'
true_label_dir = '/shared/choa/KDD_2020/true_labels/'
record_test_file = '/shared/choa/KDD_2020/RECORD-test-test-shuffled'
model_list = ['II_m{0}'.format(i) for i in range(1,17)]

def ReadLines(file, n):
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

def evaluate_ensemble_models(b, num_record_eval = 100, debug=False):
    record_test_list = ReadLines(record_test_file, num_record_eval)
    
    y_true = []
    y_scores = []
    
    for record in record_test_list:
        record = record.replace('.pickle','')
        y = np.load(true_label_dir + record + '.npy')
        y_true.extend(list(y))
        
    for i in range(len(model_list)):
        if i == len(b):
            break
        if b[i] == 1:
            if debug:
                print(model_list[i])
            y_i_scores = []
            for record in record_test_list:
                record = record.replace('.pickle','')
                yhat = np.load(pred_dir + model_list[i] + '/' + record + '.npy')
                y_i_scores.extend(list(yhat))
            y_scores.append(y_i_scores)
    
    y_scores = np.nan_to_num(np.array(y_scores))
    
    if len(y_scores) == 0:
        return 0.0
    
    if len(y_scores) > 1:
        y_scores = np.mean(y_scores,0)
    else:
        y_scores = y_scores[0]
    if debug:
        print(len(y_true), len(y_scores))
        
    return roc_auc_score(y_true, y_scores)
    
if __name__ == "__main__":
    auc = evaluate_ensemble_models(b = np.ones(12), num_record_eval = 100, debug=True)
    print(auc)
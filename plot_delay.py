from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,precision_recall_curve,auc,accuracy_score

def plot_prediction_delay():

    prob_dir = '../waveforms/'
    model_ensemble = ['II_128f_4b']
    num_per_day = 24*60*2

    early_prediction_mins = [0,1,5,10,20,30,40,50,60,120,180,240,300,360,420,480,540,600,660,720]
    early_prediction_30secs = [m*2 for m in early_prediction_mins]

    early_prediction_accs = []
    for early_30secs in early_prediction_30secs:
        y_score = []
        y_true = []
        for i in [1,2,3,4,5,6,7,9]:
            prob_list = []
            for model in model_ensemble:
                probs = np.loadtxt(prob_dir + '{0}/{0}_{1}.{2}.out'.format(i,model,1))
                prob_list.append(probs)
            
            if len(prob_list) == 1:
                prob_mean = prob_list[0]
            else:
                prob_mean = np.mean(np.array(prob_list), 0)

            if early_30secs == 0:
                y_score.extend(list(prob_mean[-(num_per_day+early_30secs):]))
            else:
                y_score.extend(list(prob_mean[-(num_per_day+early_30secs):-early_30secs]))
            y_true.extend(list(np.ones(num_per_day)))
        y_pred = (np.array(y_score) > 0.4)*1
        early_prediction_accs.append(accuracy_score(y_true, y_pred))

    plt.figure(figsize=(5,3))
    plt.plot(early_prediction_30secs, early_prediction_accs, 'o-', c='grey')
    early_prediction_hrs=[int(m/60) for m in early_prediction_mins]
    for i in range(1,8):
        early_prediction_hrs[i] = ''
    early_prediction_hrs[5] = 0.5
    plt.xticks(early_prediction_30secs, early_prediction_hrs)
    plt.xlabel('Prediction Delay in Hours', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    # plt.title('Accuracy Decreases due to Prediction Delay', fontsize=14)
    plt.tight_layout()
    plt.savefig('img/prediction_delay.png')

def plot_longer_history():

    prob_dir = '../waveforms/'
    model_ensemble = ['II_128f_4b']
    num_per_day = 24*60*2

    agg_history_mins = [0.5,1,5,10,20,30]
    agg_history_30secs = [int(m*2) for m in agg_history_mins]

    y_score = []
    agg_history_accs = []
    for agg_30secs in agg_history_30secs:
        for i in [1,2,3,4,5,6,7,9]:
            prob_list = []
            for model in model_ensemble:
                probs = np.loadtxt(prob_dir + '{0}/{0}_{1}.{2}.out'.format(i,model,1))
                prob_list.append(probs)
            
            if len(prob_list) == 1:
                prob_mean = prob_list[0]
            else:
                prob_mean = np.mean(np.array(prob_list), 0)

            cur_prob = prob_mean[-num_per_day:]
            if agg_30secs > 1:
                cur_prob = cur_prob[:int(len(cur_prob)/agg_30secs)*agg_30secs]
                cur_prob = cur_prob.reshape(-1, agg_30secs)
                cur_prob = np.mean(cur_prob, 1)
            y_score.extend(list(cur_prob))
        y_true = np.ones(len(y_score))
        y_pred = (np.array(y_score) > 0.5)*1
        agg_history_accs.append(accuracy_score(y_true, y_pred))

    plt.figure(figsize=(4,3))
    plt.plot(agg_history_30secs, agg_history_accs, 'o-', c='grey')
    plt.xticks(agg_history_30secs, agg_history_mins)
    agg_history_mins[0] = ''
    plt.xlabel('History Observed in Hours', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    # plt.title('Accuracy Increases as Longer History Observed', fontsize=14)
    plt.tight_layout()
    plt.savefig('img/longer_history.png')

def plot_longer_history_latency():
    all_res = []
    df = pd.read_csv("img/timeit_data.csv")
    all_res.append(list(df.iloc[0,:].values))
    df = pd.read_csv("img/tq_exp_obs_w_30sec.csv")
    all_res.append(list(np.mean(df.values, axis=0)))
    df = pd.read_csv("img/ts_exp_obs_w_30sec.csv")
    all_res.append(list(np.mean(df.values, axis=0)))
    df = pd.read_csv("img/total_tq_ts_exp_obs_w_30sec.csv")
    all_res.append(list(np.mean(df.values, axis=0)))
    all_res = np.array(all_res)

    plt.figure(figsize=(4,3))

    for i in range(4):
        plt.plot(all_res[i], marker=markers[i], c=colors[i], linewidth=2)

    plt.legend(['timeit', 'tq', 'ts', 'tq+ts'])
    plt.xlabel('History Observed in Hours', fontsize=14)
    plt.ylabel('Latency (seconds)', fontsize=14)
    plt.tight_layout()
    plt.savefig('img/longer_history_latency.png')

if __name__ == "__main__":

    methods = ['RD', 'GA', 'GL', 'BO', 'Ours']
    colors = ['#233D4D', 'tab:gray', '#48A9A6', '#2F6690', '#F9592C']
    markers = ['v', 'd', 's', 'o']
    # colors = ['#247BA0', '#70C1B3', '#B2DBBF', '#F3FFBD', '#FF1654']

    # plot_prediction_delay()

    # plot_longer_history()

    # plot_longer_history_latency()
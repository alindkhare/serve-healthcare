from matplotlib import pyplot as plt
import numpy as np

def read_traj(traj_fname):

    log_method = ['solve_opt_passive', 'solve_proxy']

    # read traj
    traj_method = []
    traj_accuracy = []
    traj_latency = []
    with open(traj_fname, 'r') as fin:
        fin.readline()
        for line in fin:
            content = line.strip().split(',')
            method = content[3]
            accuracy = np.array([float(i) for i in content[4:6]])
            latency = float(content[6])
            traj_method.append(method)
            traj_accuracy.append(accuracy)
            traj_latency.append(latency)
    traj_method = np.array(traj_method)
    traj_accuracy = np.array(traj_accuracy)
    traj_latency = np.array(traj_latency)
    out_accuracy = []
    out_latency = []
    for m in log_method:
        out_accuracy.append(traj_accuracy[traj_method==m])
        out_latency.append(traj_latency[traj_method==m])

    return out_accuracy, out_latency

if __name__ == "__main__":

    traj_fname = 'res/finished/traj_20200212_194247_60models_latency0.15.txt'
    out_accuracy, out_latency = read_traj(traj_fname)
    fig = plt.figure(figsize=(2,3))    
    accuracy_npo = out_accuracy[0][:,0]
    latency_npo = out_latency[0]
    accuracy_npo = accuracy_npo[latency_npo < 0.15]
    accuracy_npo = accuracy_npo[accuracy_npo != 0]
    accuracy_ours = out_accuracy[1][:,0]
    latency_ours = out_latency[1]
    accuracy_ours = accuracy_ours[latency_ours < 0.15]
    accuracy_ours = accuracy_ours[accuracy_ours != 0]
    plt.boxplot([accuracy_npo, accuracy_ours], widths=0.75)
    plt.ylabel('ROC-AUC')
    plt.xlabel(r'Latency $\leq$ 0.15 s')
    plt.xticks([1,2], ('NPO', 'HOLMES'))
    plt.tight_layout()
    fig.savefig('img/accuracy_box_1.pdf')

    traj_fname = 'res/finished/traj_20200211_232156_60models_latency0.2.txt'
    out_accuracy, out_latency = read_traj(traj_fname)
    fig = plt.figure(figsize=(2,3))    
    accuracy_npo = out_accuracy[0][:,0]
    latency_npo = out_latency[0]
    accuracy_npo = accuracy_npo[latency_npo < 0.2]
    accuracy_npo = accuracy_npo[accuracy_npo != 0]
    accuracy_ours = out_accuracy[1][:,0]
    latency_ours = out_latency[1]
    accuracy_ours = accuracy_ours[latency_ours < 0.2]
    accuracy_ours = accuracy_ours[accuracy_ours != 0]
    plt.boxplot([accuracy_npo, accuracy_ours], widths=0.75)
    plt.ylabel('ROC-AUC')
    plt.xlabel(r'Latency $\leq$ 0.20 s')
    plt.xticks([1,2], ('NPO', 'HOLMES'))
    plt.tight_layout()
    fig.savefig('img/accuracy_box_2.pdf')

    traj_fname = 'res/finished/traj_20200212_231319_60models_latency0.25.txt'
    out_accuracy, out_latency = read_traj(traj_fname)
    fig = plt.figure(figsize=(2,3))    
    accuracy_npo = out_accuracy[0][:,0]
    latency_npo = out_latency[0]
    accuracy_npo = accuracy_npo[latency_npo < 0.25]
    accuracy_npo = accuracy_npo[accuracy_npo != 0]
    accuracy_ours = out_accuracy[1][:,0]
    latency_ours = out_latency[1]
    accuracy_ours = accuracy_ours[latency_ours < 0.25]
    accuracy_ours = accuracy_ours[accuracy_ours != 0]
    plt.boxplot([accuracy_npo, accuracy_ours], widths=0.75)
    plt.ylabel('ROC-AUC')
    plt.xlabel(r'Latency $\leq$ 0.25 s')
    plt.xticks([1,2], ('NPO', 'HOLMES'))
    plt.tight_layout()
    fig.savefig('img/accuracy_box_3.pdf')

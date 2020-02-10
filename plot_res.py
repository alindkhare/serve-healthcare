from matplotlib import pyplot as plt
import numpy as np

def read_res(log_fname, traj_fname):
    # read log
    log_method = []
    log_accuracy = []
    log_latency = []
    with open(log_fname, 'r') as fin:
        fin.readline()
        for line in fin:
            content = line.strip().split(',')
            method = content[2]
            accuracy = np.array([float(i) for i in content[3:15]])
            latency = float(content[15])
            log_method.append(method)
            log_accuracy.append(accuracy)
            log_latency.append(latency)
    log_accuracy = np.array(log_accuracy)
    log_latency = np.array(log_latency)

    # read traj
    traj_method = []
    traj_accuracy = []
    traj_latency = []
    with open(traj_fname, 'r') as fin:
        fin.readline()
        for line in fin:
            content = line.strip().split(',')
            method = content[2]
            accuracy = np.array([float(i) for i in content[3:15]])
            latency = float(content[15])
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

    return log_accuracy, log_latency, out_accuracy, out_latency

def plot_fig1_intro():
    fig, ax = plt.subplots(1,2,figsize=(8,3))
    ax[1].bar(methods, log_accuracy[:,2], color=colors)
    ax[1].set_ylim([0.925,0.96])
    ax[1].set_ylabel('Accuracy (PR-AUC)')
    ax[0].bar(methods, log_latency, color=colors)
    ax[0].set_ylim([0.4,0.6])
    ax[0].set_ylabel('Latency (Seconds)')
    plt.tight_layout()
    plt.savefig('img/intro.png')


if __name__ == "__main__":

    methods = ['RD', 'GA', 'GL', 'BO', 'Ours']
    colors = ['#233D4D', 'tab:gray', '#48A9A6', '#2F6690', '#F9592C']

    log_fname = 'res/log_20200210_172950.txt'
    traj_fname = 'res/traj_20200210_172950.txt'
    log_accuracy, log_latency, traj_accuracy, traj_latency = read_res(log_fname, traj_fname)
    print(traj_latency)

    plot_fig1_intro()
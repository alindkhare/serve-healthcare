from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,precision_recall_curve,auc,accuracy_score

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

    return log_accuracy, log_latency, out_accuracy, out_latency

def plot_fig1_intro():
    """
    deprecated
    """
    fig, ax = plt.subplots(1,2,figsize=(8,3))
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[0].bar(methods, log_latency, color=colors)
    ax[0].set_ylabel('Latency (Seconds)')
    # ax[0].set_yticks(np.arange(0, 1, 0.04))
    ax[0].set_ylim([0.12,0.2])
    ax[1].grid()
    ax[1].set_axisbelow(True)
    ax[1].bar(methods, log_accuracy[:,0], color=colors)
    # ax[1].set_yticks(np.arange(0, 1, 0.01))
    ax[1].set_ylim([0.86,0.96])
    ax[1].set_ylabel('ROC-AUC')
    plt.tight_layout()
    plt.savefig('img/intro.png')

def get_traj(x, y, m):
    out_x = []
    out_y = []
    if m in [0,1,2]:
        for i in range(1,len(x)+1):
            idx = np.argmax(y[:i])
            out_x.append(x[idx])
            out_y.append(y[idx])
        if m == 0:
            out_x.append(0.26995220)
            out_y.append(0.8871)
        if m == 1:
            out_x.append(0.31019037)
            out_y.append(0.9385)
        if m == 2:
            out_x.append(0.22935618)
            out_y.append(0.9019)
        return np.array(out_x), np.array(out_y)
    if m in [3]:
        for i in range(1,len(x)+1,10):
            idx = np.argmax(y[:i])
            out_x.append(x[idx])
            out_y.append(y[idx])
        return np.array(out_x), np.array(out_y)
    if m in [4]:
        return x[:-1], y[:-1]

def plot_fig4_fig5_explore():
    # all_accuracy = []
    # with open('bk/cache_accuracy_small.txt', 'r') as fin:
    #     for line in fin:
    #         content = line.strip().split('|')
    #         accuracy = np.array([float(i.strip()) for i in content[2].replace('(', '').replace(')', '').split(',')])
    #         all_accuracy.append(accuracy)
    # all_accuracy = np.array(all_accuracy)

    # all_latency = []
    # with open('bk/cache_latency_small.txt', 'r') as fin:
    #     for line in fin:
    #         content = line.strip().split('|')
    #         latency = float(content[2])
    #         all_latency.append(latency)
    # all_latency = np.array(all_latency)

    plt.figure(figsize=(4,3))
    # plt.scatter(all_latency, all_accuracy[:,0], c='lightgrey')
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xlim([0,0.2])
    plt.grid()
    for i in range(5):
        plot_traj_x, plot_traj_y = get_traj(traj_latency[i], traj_accuracy[i][:,0], m=i)
        plt.scatter(plot_traj_x, plot_traj_y, c=colors[i])
        plt.plot(plot_traj_x, plot_traj_y, marker=markers[i], c=colors[i], linewidth=2)
    plt.legend(methods)
    plt.xlabel('Latency (seconds)')
    plt.ylabel('ROC-AUC')
    plt.tight_layout()
    plt.savefig('img/explore.pdf')

    plt.figure(figsize=(4,3))
    plt.grid()
    for i in range(5):
        plot_traj_x, plot_traj_y = get_traj(traj_latency[i], traj_accuracy[i][:,0], m=i)
        plt.plot(plot_traj_x, marker=markers[i], c=colors[i], linewidth=2)
    plt.axhline(0.2, c='r', linestyle='--')
    # plt.text(x=3, y=0.21, s='Latency Constraint', fontsize=14, c='r')
    plt.legend(methods, ncol=3)
    plt.xlabel('Number of Explorations')
    plt.ylabel('Latency (seconds)')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.ylim([0.05,0.4])
    plt.tight_layout()
    plt.savefig('img/explore_latency.pdf')

    plt.figure(figsize=(4,3))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.ylim([0.8,0.97])
    plt.grid()
    for i in range(5):
        plot_traj_x, plot_traj_y = get_traj(traj_latency[i], traj_accuracy[i][:,0], m=i)
        plt.plot(plot_traj_y, marker=markers[i], c=colors[i], linewidth=2)
    plt.legend(methods, ncol=2)
    plt.xlabel('Number of Explorations')
    plt.ylabel('ROC-AUC')
    plt.tight_layout()
    plt.savefig('img/explore_accuracy.pdf')


def plot_fig1_intro_new():
    """
    """
    # rotation = 20
    # tags = ['Random', 'Best Accuracy', 'Best Latency', 'HOLMES']
    tags = ['Best\nAccuracy', 'HOLMES', 'Random', 'Best\nLatency']
    colors = ['tab:gray', '#F9592C', '#233D4D', '#48A9A6']
    log_latency = [0.91436863, 0.18869656, 0.36, 0.056254166]
    log_accuracy = [0.9439, 0.925, 0.8758, 0.862784727]
    fig, ax = plt.subplots(1,2,figsize=(8,3))
    ax[1].grid()
    ax[1].set_axisbelow(True)
    ax[1].bar(tags, log_latency, color=colors)
    ax[1].set_ylabel('Latency (Seconds)')
    # ax[1].set_xticklabels(tags, rotation=rotation)
    # ax[1].set_yticks(np.arange(0, 1, 0.04))
    # ax[1].set_ylim([0.12,0.2])
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[0].bar(tags, log_accuracy, color=colors)
    # ax[0].set_yticks(np.arange(0, 1, 0.01))
    ax[0].set_ylim([0.85,0.96])
    ax[0].set_ylabel('ROC-AUC')
    # ax[0].set_xticklabels(tags, rotation=rotation)
    plt.tight_layout()
    plt.savefig('img/intro.pdf')

if __name__ == "__main__":

    methods = ['RD', 'AF', 'LF', 'NPO', 'HOLMES']
    colors = ['#233D4D', 'tab:gray', '#48A9A6', '#2F6690', '#F9592C']
    # colors = ['#247BA0', '#70C1B3', '#B2DBBF', '#F3FFBD', '#FF1654']
    markers = ['v', '^', 'd', 's', 'o']

    log_fname = 'res/finished/log_20200211_232156_60models_latency0.2.txt'
    traj_fname = 'res/finished/traj_20200211_232156_60models_latency0.2.txt'
    log_accuracy, log_latency, traj_accuracy, traj_latency = read_res(log_fname, traj_fname)

    plot_fig1_intro_new()

    # plot_fig4_fig5_explore()

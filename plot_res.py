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
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[0].bar(methods, log_latency, color=colors)
    ax[0].set_ylabel('Latency (Seconds)')
    ax[0].set_yticks(np.arange(0, 1, 0.04))
    ax[0].set_ylim([0.4,0.6])
    ax[1].grid()
    ax[1].set_axisbelow(True)
    ax[1].bar(methods, log_accuracy[:,0], color=colors)
    ax[1].set_yticks(np.arange(0, 1, 0.01))
    ax[1].set_ylim([0.955,0.98])
    ax[1].set_ylabel('Accuracy (ROC-AUC)')
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
        return np.array(out_x), np.array(out_y)
    if m in [3]:
        for i in range(1,len(x)+1,2):
            idx = np.argmax(y[:i])
            out_x.append(x[idx])
            out_y.append(y[idx])
        return np.array(out_x), np.array(out_y)
    
    else:
        return x, y

def plot_fig4_fig5_explore():
    all_accuracy = []
    with open('cache_accuracy_small.txt', 'r') as fin:
        for line in fin:
            content = line.strip().split('|')
            accuracy = np.array([float(i.strip()) for i in content[2].replace('(', '').replace(')', '').split(',')])
            all_accuracy.append(accuracy)
    all_accuracy = np.array(all_accuracy)

    all_latency = []
    with open('cache_latency_small.txt', 'r') as fin:
        for line in fin:
            content = line.strip().split('|')
            latency = float(content[2])
            all_latency.append(latency)
    all_latency = np.array(all_latency)

    plt.figure(figsize=(4,3))
    plt.scatter(all_latency, all_accuracy[:,0], c='lightgrey')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid()
    for i in range(5):
        plot_traj_x, plot_traj_y = get_traj(traj_latency[i], traj_accuracy[i][:,0], m=i)
        plt.scatter(plot_traj_x, plot_traj_y, c=colors[i])
        plt.plot(plot_traj_x, plot_traj_y, c=colors[i], linewidth=2)
    plt.legend(methods)
    plt.xlabel('Latency')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('img/explore.png')

    plt.figure(figsize=(4,3))
    plt.yticks(np.arange(0, 1, 0.2))
    plt.grid()
    for i in range(5):
        plot_traj_x, plot_traj_y = get_traj(traj_latency[i], traj_accuracy[i][:,0], m=i)
        plt.plot(plot_traj_x, c=colors[i], linewidth=2)
    plt.legend(methods)
    plt.xlabel('Number of Explore')
    plt.ylabel('Latency')
    plt.tight_layout()
    plt.savefig('img/explore_latency.png')

    plt.figure(figsize=(4,3))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid()
    for i in range(5):
        plot_traj_x, plot_traj_y = get_traj(traj_latency[i], traj_accuracy[i][:,0], m=i)
        plt.plot(plot_traj_y, c=colors[i], linewidth=2)
    plt.legend(methods)
    plt.xlabel('Number of Explore')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('img/explore_accuracy.png')

if __name__ == "__main__":

    methods = ['RD', 'GA', 'GL', 'BO', 'Ours']
    colors = ['#233D4D', 'tab:gray', '#48A9A6', '#2F6690', '#F9592C']
    # colors = ['#247BA0', '#70C1B3', '#B2DBBF', '#F3FFBD', '#FF1654']

    log_fname = 'res/log_20200210_172950.txt'
    traj_fname = 'res/traj_20200210_172950.txt'
    log_accuracy, log_latency, traj_accuracy, traj_latency = read_res(log_fname, traj_fname)

    plot_fig1_intro()

    plot_fig4_fig5_explore()

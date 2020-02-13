from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":

    colors = ['#233D4D', 'tab:gray', '#48A9A6', '#2F6690', '#F9592C']
    markers = ['v', '^', 'd', 's', 'o']
    methods = ['Accuracy Surrogate', 'Latency Surrogate']

    data = pd.read_csv('res/finished/proxy_20200212_231319_60models_latency0.25.txt').values
    print(data)

    # 1
    plt.figure(figsize=(4,3))
    plt.grid()
    i = 0
    plt.plot(data[1:,i], marker=markers[i], c=colors[i], linewidth=2)
    i = 2
    plt.plot(data[1:,i], marker=markers[i], c=colors[i], linewidth=2)
    plt.legend(methods)
    plt.xlabel('Number of Explorations')
    plt.ylabel('MAE')
    plt.tight_layout()
    plt.savefig('img/fit_mae.pdf')

    # 1
    plt.figure(figsize=(4,3))
    plt.grid()
    i = 1
    plt.plot(data[1:,i], marker=markers[i-1], c=colors[i-1], linewidth=2)
    i = 3
    plt.plot(data[1:,i], marker=markers[i-1], c=colors[i-1], linewidth=2)
    plt.legend(methods)
    plt.xlabel('Number of Explorations')
    plt.ylabel('R2 Score')
    plt.tight_layout()
    plt.savefig('img/fit_r2.pdf')

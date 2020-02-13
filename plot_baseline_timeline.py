import argparse
import jsonlines
import json
# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', nargs='+', type=str, default='profile_results.jsonl', help='start system with test config')
parser.add_argument('-p', '--path', type=str, default='profiled_examples', help='path to profile pdf image results')
args = parser.parse_args()
color = ['#F9592C', '#233D4D']
holmes_latency = 0.19980348
batch_latency = 2.09

def _calculate_latency(file_name):
    #print(file_name)
    latency_s = []
    with jsonlines.open(file_name) as reader:
        print(reader)
        latency_s = [(obj["end"] - obj["start"]) for obj in reader]
    #print(latency_s.sort())
    return latency_s

def draw_latency_trace1(list_files):
    print('draw latency trace...')
    labels = []
    plt.clf()
    fig = plt.figure(figsize=(20,2))
    all_df = []
    for color_idx,input_file in enumerate(list_files):
        fname = input_file.split(".")[0]
        labels.append("HOLMES")
        minute_latency =_calculate_latency(input_file)
        #print(minute_latency)
        minute_dc = minute_latency[:100]
        minute_l = holmes_latency
        hour_latency = []
        for i in range(60):
            for dc in minute_dc:
                hour_latency.append(dc)
            hour_latency.append(minute_l)
        print(hour_latency)
        #print("max {} is {}".format(fname, max(latency)))
        df = pd.DataFrame(hour_latency, columns=['latency(s)'])
        print("hour_latency {} is {} ln {}".format(fname, len(hour_latency), len(df)))
        # plt.plot(df.index, df["latency(s)"], c=color[color_idx])
        #draw batching
        min_val = min(hour_latency)
        batching_hour_latency = [min_val for i in range(len(hour_latency))]
        batching_hour_latency[-1] = batch_latency
        #print("max {} is {}".format(fname, max(latency)))
        df = pd.DataFrame(batching_hour_latency, columns=['latency(s)'])
        print("batching_hour_latency {} is {} ln {}".format(fname, len(batching_hour_latency), len(df)))
        labels.append("Batching")
        plt.plot(df.index, df["latency(s)"], c=color[color_idx+1])

    # plt.yscale('log')

    #final_df = pd.concat(all_df)
    # lgd = plt.legend(labels, fontsize=14, loc='upper left')
    #x = [i * 3750 for i in range(12)]
    # plt.ylim([0,25])
    locs, labels = plt.xticks()
    print(locs)
    print(labels)
    labels = ("", "0", "10", "20", "30", "40", "50", "60", "")
    plt.xticks(locs, labels, fontsize=12)
    plt.xlim([-200,6200])
    
    #plt.axhline(y=1, color='r', linestyle='-')
    #plt.yscale('log')
    plt.xlabel("Timeline (minutes)", fontsize=14)
    plt.ylabel("Latency (seconds)", fontsize=14)
    fig.savefig('img/queue_id_to_latency_time_1.png', bbox_inches='tight')
    plt.show()
    
def draw_latency_trace2(list_files):
    print('draw latency trace...')
    labels = []
    plt.clf()
    fig = plt.figure(figsize=(20,2))
    all_df = []
    for color_idx,input_file in enumerate(list_files):
        fname = input_file.split(".")[0]
        labels.append("HOLMES")
        minute_latency =_calculate_latency(input_file)
        #print(minute_latency)
        minute_dc = minute_latency[:100]
        minute_l = holmes_latency
        hour_latency = []
        for i in range(60):
            for dc in minute_dc:
                hour_latency.append(dc)
            hour_latency.append(minute_l)
        print(hour_latency)
        #print("max {} is {}".format(fname, max(latency)))
        df = pd.DataFrame(hour_latency, columns=['latency(s)'])
        print("hour_latency {} is {} ln {}".format(fname, len(hour_latency), len(df)))
        plt.plot(df.index, df["latency(s)"], c=color[color_idx])
        #draw batching
        min_val = min(hour_latency)
        batching_hour_latency = [min_val for i in range(len(hour_latency))]
        batching_hour_latency[-1] = batch_latency
        #print("max {} is {}".format(fname, max(latency)))
        df = pd.DataFrame(batching_hour_latency, columns=['latency(s)'])
        print("batching_hour_latency {} is {} ln {}".format(fname, len(batching_hour_latency), len(df)))
        labels.append("Batching")
        # plt.plot(df.index, df["latency(s)"], c=color[color_idx+1])

    # plt.yscale('log')

    #final_df = pd.concat(all_df)
    # lgd = plt.legend(labels, fontsize=14, loc='upper left')
    #x = [i * 3750 for i in range(12)]
    # plt.ylim([0,25])
    locs, labels = plt.xticks()
    print(locs)
    print(labels)
    labels = ("", "0", "10", "20", "30", "40", "50", "60", "")
    plt.xticks(locs, labels, fontsize=12)
    plt.yticks([0.01,0.1,1,10])
    plt.xlim([-200,6200])
    
    #plt.axhline(y=1, color='r', linestyle='-')
    #plt.yscale('log')
    plt.xlabel("Timeline (minutes)", fontsize=14)
    plt.ylabel("Latency (seconds)", fontsize=14)
    fig.savefig('img/queue_id_to_latency_time_2.png', bbox_inches='tight')
    plt.show()
    


if __name__ == '__main__':
    filename = ['img/solve_proxy_profile_results.jsonl']

    draw_latency_trace1(filename)
    draw_latency_trace2(filename)

import csv

import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np

out_path = './out/'

x = range(4)
name = ['ONES', 'DRL', 'TIRESIAS', 'OPTIMUS']
color = ['firebrick', 'lightsteelblue', 'steelblue', 'mediumaquamarine']
hatch = ['/', '\\', '-', '+']
line = ['-', '--', ':', '-.']

jct = [[] for _ in x]
exec = [[] for _ in x]
delay = [[] for _ in x]


def read_results(i):
    with open(out_path + name[i] + '_profile.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for id, j, e, d in reader:
            jct[i].append(int(j))
            exec[i].append(int(e))
            delay[i].append(int(d))


def plot_results(data, res_type):
    f = plt.figure(figsize=[5, 3.75])
    for i in x:
        y = np.mean(data[i])
        plt.bar(x[i],
                y,
                width=0.5,
                ec='k',
                hatch=hatch[i],
                color=color[i],
                zorder=2)
        plt.text(x[i], y, '%.2f' % y, ha='center', va='bottom')
    plt.xticks(x, name)
    plt.grid(axis='y', zorder=1)
    plt.ylabel('Average ' + res_type + ' (s)')
    f.savefig(res_type + '.pdf', bbox_inches='tight')

    f = plt.figure(figsize=[5, 3.75])

    plt.boxplot(data,
                positions=x,
                showfliers=False,
                labels=name,
                widths=0.6,
                zorder=2)
    plt.xticks(x, name)
    plt.grid(axis='y', zorder=1)
    plt.ylabel(res_type + ' (s)')
    plt.legend(loc='lower right')
    f.savefig(res_type + '-box.pdf', bbox_inches='tight')

    f = plt.figure(figsize=[5, 3.75])
    for i in x:
        z = np.sort(data[i])
        y = [(j + 1) / len(data[i]) for j in range(len(data[i]))]
        plt.plot(z, y, color=color[i], ls=line[i], linewidth=3, label=name[i])
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xscale('log')
    plt.xlabel(res_type + ' (s)')
    plt.ylabel('CF')
    plt.grid()
    plt.legend(loc='lower right')
    f.savefig(res_type + '-cdf.pdf', bbox_inches='tight')
    plt.close()


for i in x:
    read_results(i)

plot_results(jct, 'JCT')
plot_results(exec, 'execution_time')
plot_results(delay, 'queuing_time')

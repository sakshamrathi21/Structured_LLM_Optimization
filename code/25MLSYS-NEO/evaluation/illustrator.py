import json
import os
import matplotlib.pyplot as plt
import numpy as np

cur_dir = os.path.dirname(os.path.realpath(__file__))

def get_lat_avg(file):
    with open(file) as f:
        data = json.load(f)
    # only take latter half
    data = data[len(data) // 4:]
    return sum([(x['end'] - x['start']) / (x['output_len']) for x in data]) / len(data)


def draw_one_rl_diagram(
    title: str,
    data_name: str,
    sys_file_names: list[str],
    sys_legend_names: list[str],
    rate_lists: list[list[float]],
    ylim: float,
    markers: list[str],
    set_ylabel: bool = False,
):
    lats = []
    max_rate = max([max(rate_list) for rate_list in rate_lists])
    for sys_file_name, rate_list in zip(sys_file_names, rate_lists):
        lats.append([])
        for rate in rate_list:
            rate_str = str(rate).replace(".", "_")
            lats[-1].append(get_lat_avg(f"{cur_dir}/results/{sys_file_name}-{data_name}-lat-{rate_str}.json"))

    # ax.set_title(title, y=-0.3, fontsize="x-large")

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i, sys_legend_name in enumerate(sys_legend_names):
        ax.plot(rate_lists[i], lats[i], label=sys_legend_name, marker=markers[i])

    ax.set_xlabel("Ruquest rate (req/s)", fontsize="large")
    if set_ylabel:
        ax.set_ylabel("Average per token latency (s)", fontsize="large")
    ax.set_xlim(0, max_rate)
    ax.set_xticks([0.5 * x for x in range(round(max_rate * 2 + 1))])
    ax.set_ylim(-ylim / 50, ylim)
    ax.set_yticks([ylim / 5 * x for x in range(6)])
    ax.set_xticklabels([f"{x:.1f}" for x in ax.get_xticks()], fontsize="large")
    ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()], fontsize="large")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend()
    plt.savefig(f"{cur_dir}/{title}.pdf", bbox_inches='tight')
    return handles, labels


def get_tp(filenames: list[str], interv: tuple[float, float]):
    tps = []
    for i, filename in enumerate(filenames):
        with open(filename) as f:
            data = json.load(f)

        times = sorted([d['end'] for d in data])
        data = [times[j] - times[j-1] for j in range(1, len(times))]

        ndata = len(data)
        nwarmup = round((ndata + 1) * interv[0])
        ncooldown = round((ndata + 1) * interv[1])
        tps.append(1 / np.mean(data[nwarmup: ncooldown]))
    return tps

def get_tp_token(filenames: list[str]):
    tps = []
    for i, filename in enumerate(filenames):
        with open(filename) as f:
            data = json.load(f)

        first_start = min([d['start'] for d in data])
        last_end = max([d['end'] for d in data])
        total_time = last_end - first_start
        total_tokens = sum([d['output_len'] + d['input_len'] for d in data])
        tps.append(total_tokens / total_time)

    return tps


def draw_one_ps_diagram(
    title: str,
    base_sys_name: str,
    interv: list[float],
    num_datas: list[int],
    sys_file_names: list[str],
    legend_names: list[str | None],
    input_lens: list[int],
    output_lens: list[int],
    markers: list[str],
    show_ylabels: bool = False,
    show_legend: bool = True,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i in range(len(num_datas)):
        tps = []
        for out_len in output_lens:
            file_names = [f'{cur_dir}/results/{sys_name}-{num_datas[i]}-{input_lens[i]}-{out_len}-tp.json' for sys_name in [base_sys_name, sys_file_names[i]]]
            tp_pair = get_tp(file_names, interv)
            tps.append(tp_pair)
               
        ratios = [tp1 / tp0 for tp0, tp1 in tps]
        ax.plot(output_lens, ratios, label=f'{legend_names[i]}', marker=markers[i])

    # draw y = 1 line
    ax.plot([output_lens[0], output_lens[-1]], [1, 1], 'r--', label='baseline')

    ax.set_xlabel('Avg. output length', fontsize='large')
    if show_ylabels:
        ax.set_ylabel('Relative throughput', fontsize='large')
    ax.set_xticklabels([f'{x:.0f}' for x in ax.get_xticks()], fontsize='large')
    ax.set_yticklabels([f'{x:.2f}' for x in ax.get_yticks()], fontsize='large')
    handles, labels = ax.get_legend_handles_labels()
    if show_legend:
        ax.legend()
    fig.savefig(f'{cur_dir}/{title}.pdf', bbox_inches='tight')
    return handles, labels


def parse_ours_server_log(file):
    # Get sizes list from lines like below
    # INFO:swiftllm.server.engine:Forwarding batches with sizes [(0, 1, 14, 0)], swap out: 0, swap in: 4
    with open(file) as f:
        lines = f.readlines()

    sizes = []
    for line in lines:
        if 'Forwarding batches with sizes' in line:
            sizes.append(eval(line.split('[')[1].split(']')[0]))

    return sizes


def parse_vllm_server_log(file):
    # Get Rumnings list from lines like below
    # INFO 10-28 08:28:24 metrics.py:351] Avg prompt throughput: 5806.9 tokens/s, Avg generation throughput: 53.0 tokens/s, Running: 12 reqs, Swapped: 0 reqs, Pending: 1466 reqs, GPU KV cache usage: 93.6%, CPU KV cache usage: 0.0%.
    with open(file) as f:
        lines = f.readlines()

    runnings = []
    for line in lines:
        if 'Running' in line and 'Avg prompt throughput' in line:
            runnings.append(int(line.split('Running: ')[1].split(' reqs')[0]))

    return runnings

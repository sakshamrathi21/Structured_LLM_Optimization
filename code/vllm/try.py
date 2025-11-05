# from nanovllm import LLM, SamplingParams
# import os

# import random
# import string
# import time
# import sys as system

# from transformers import AutoTokenizer
# from system_prompt import queries, system_prompt


# def generate_prefixes(n):
#     letters = string.ascii_uppercase
#     prefixes = []
#     for a in letters:
#         for b in letters:
#             prefixes.append(a + b)
#     return prefixes[:n]


# def main():
#     model_name = "meta-llama/Llama-3.1-8B-Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     num_families = int(system.argv[1]) if len(system.argv) > 1 else 10
#     max_num_seqs = int(system.argv[2]) if len(system.argv) > 2 else 100
#     whether_random = int(system.argv[3]) if len(system.argv) > 3 else 1
#     prefixes = generate_prefixes(num_families)  # 10 families
#     prompts = []
#     families = []  

#     for prefix in prefixes: 
#         for q in queries:
#             modified_system_prompt = prefix + " " + system_prompt
#             chat_messages = [
#                 {"role": "system", "content": modified_system_prompt},
#                 {"role": "user", "content": q},
#             ]
#             try:
#                 full_prompt = tokenizer.apply_chat_template(
#                     chat_messages,
#                     tokenize=False,
#                     add_generation_prompt=True,
#                 )
#             except Exception:
#                 full_prompt = modified_system_prompt + " Question: " + q

#             prompts.append(full_prompt)
#             families.append(prefix)  # <-- mark which family it belongs to

#     print(f"Generated {len(prompts)} prompts in total.")

#     enable_chunked_prefill = True
#     model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/")
#     llm = LLM(model_path, tensor_parallel_size=1, max_num_seqs=max_num_seqs)
#     paired = list(zip(prompts, families))
#     if whether_random:
#         random.shuffle(paired)
#     prompts, families = zip(*paired)
#     prompts = list(prompts)
#     families = list(families)
#     sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
#     print("Total number of requests to simulate:", len(prompts), "for max_num_seqs =", max_num_seqs)
#     initial_time = time.time()
#     print("Simulation started at time:", initial_time)
#     outputs = llm.generate(list(prompts), sampling_params)
#     print("Simulation completed at time:", time.time())
#     print("Throughput of vLLM:", len(outputs) / (time.time() - initial_time), "req/s", "for number of families =", num_families, "and max_num_seqs =", max_num_seqs, "whether random =", whether_random)
#     # print(outputs)

# if __name__ == "__main__":
#     main()

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- READ AND PARSE ----------
data = []
pattern = re.compile(
    r"Throughput Test with (\d+) families, (\d+) req/sec and max_batch_size of (\d+) \(randomness: (\d+)\): ([\d\.]+)"
)

tuple_to_throughput = {}

with open("vllm_multiple_families_varying_workloads_output.txt") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            families, req, batch, rand, value = m.groups()
            tuple_to_throughput[(int(families), int(req), int(batch), int(rand))] = float(value)
for (key, value) in tuple_to_throughput.items():
    families, req, batch, rand = key
    data.append({
        "families": families,
        "req_per_sec": req,
        "batch_size": batch,
        "randomness": rand,
        "throughput": value
    })

df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ---------- PLOT: Separate for each number of families ----------
for families, df_fam in df.groupby("families"):
    plt.figure(figsize=(15, 10))
    groups = list(df_fam.groupby(["batch_size", "randomness"]))
    colors = cm.get_cmap('tab20', len(groups))  # or try 'husl', 'viridis', etc.

    for idx, ((batch, rand), group) in enumerate(groups):
        group = group.sort_values("req_per_sec")
        plt.plot(
            group["req_per_sec"],
            group["throughput"],
            marker="o",
            color=colors(idx),
            label=f"Batch={batch}, Rand={rand}"
        )

    plt.xlabel("Requests per Second")
    plt.ylabel("Throughput")
    plt.title(f"Throughput vs Requests per Second (Families = {families})")
    plt.grid(True)

    # ---- Legend outside plot ----
    plt.legend(
        bbox_to_anchor=(1.05, 1),  # shift legend to right
        loc='upper left',
        borderaxespad=0.
    )

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave room for legend
    plt.show()
    plt.savefig(f"../../results/throughput_vs_req_batch_rand_families_{families}.png")


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="batch_size", y="throughput", hue="randomness")
plt.title("Throughput Distribution Across Randomness Levels")
plt.show()
plt.savefig("../../results/throughput_boxplot_batch_size_randomness.png")
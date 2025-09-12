"""
Reproduction script for Figure 10a in the paper.

Please note that this script should be run on g5.x16large instances, and only the corresponding line
is drew in the figure. 

The full test takes a long time (~5h) to run. You can reduce the number of requests or number of data
points to speed up the evaluation process.
"""

import asyncio
import json
import os

from server import start_server, stop_server
from benchmark import run_test, prepare_mock_test
from illustrator import draw_one_ps_diagram


# Tweak hyperparameters here:

num_data = 2000
# Number of total request send to the serving engine, reduce this number to speed up the evaluation process. 
# However, the result may not be as accurate as the original one due to warm-up and cool-down effects. It is 
# not recommended to set this number below 800.

input_len = 1000
# Length of input sequence, please keep it as 1000 to reproduce the original result.

output_lens = [50, 100, 200, 300, 400]
# Length of output sequence, reduce the number of elements in the list to speed up the evaluation process.


cur_dir = os.path.dirname(os.path.realpath(__file__))
with open(f"{cur_dir}/configs/config-a10-8b.json", "r") as f:
    config = json.load(f)


async def one_round(server_name: str):
    start_server(server_name, config)
    try:
        for output_len in output_lens:
            await run_test(*prepare_mock_test(num_data, input_len, output_len, server_name, config))
    finally:
        stop_server()
    await asyncio.sleep(5)


async def main():
    await one_round("base")
    await one_round("ours")


if __name__ == "__main__":
    asyncio.run(main())
    draw_one_ps_diagram(
        title="fig10a",
        base_sys_name="base",
        interv=[0.3, 0.7], # The interval for calculating throughput, we ignore the first 30% and last 30% of the data in order to avoid warm-up and cool-down effects.
        num_datas=[num_data],
        sys_file_names=["ours"],
        legend_names=["x16large"],
        input_lens=[input_len],
        output_lens=output_lens,
        markers=["x"],
        show_ylabels=True,
        show_legend=True
    )

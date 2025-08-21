from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

# This is equivalent to running the following command in your terminal

# python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --mem-fraction-static 0.7 --disable-cuda-graph

# python3 -m sglang.launch_server   --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0   --host 0.0.0.0

server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
 --host 0.0.0.0 --mem-fraction-static 0.7 --disable-cuda-graph
"""
)

wait_for_server(f"http://localhost:{port}")

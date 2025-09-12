import os
import sys
import subprocess
import time
import logging

cur_dir = os.path.dirname(os.path.abspath(__file__))
neo_dir = os.path.dirname(cur_dir)
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{cur_dir}/evaluation.log", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

server_proc = None

def start_server(name: str, config: dict):
    """
    Start the server
    """
    # pylint: disable=global-statement
    global server_proc

    numacmd = ["numactl", "-N", "0", "-m", "0"]
    with open(f"{cur_dir}/{name}-server.log", "w") as f:
        if name[:4] == "vllm":
            chunk_size_str = name[4:] if name != "vllm" else str(config["num_gpu_blocks_override"] * config["block_size"])
            max_num_seqs = min(int(chunk_size_str), config["max_num_seqs"])
            server_proc = subprocess.Popen(
                numacmd + [
                    "vllm", "serve", config["model_path"], "--port", "8000",
                    "--block-size", str(config["block_size"]),
                    "--max-model-len", str(config["max_model_len"]),
                    "--max-num-seqs", str(max_num_seqs),
                    "--max-num-batched-tokens", chunk_size_str,
                    "--tensor-parallel-size", str(config["tensor_parallel_size"]),
                    # "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
                    "--num-gpu-blocks-override", str(config["num_gpu_blocks_override"]),
                    "--swap-space", str(config["swap_space"] / config["tensor_parallel_size"]),
                    "--enforce-eager",
                    "--disable-sliding-window",
                    "--disable-async-output-proc",
                    "--disable-custom-all-reduce",
                    "--disable-frontend-multiprocessing",
                    "--tokenizer-pool-size", "1",
                    "--enable-chunked-prefill",
                    "--preemption-mode", "recompute",
                    "--dtype", "float16"
                ], 
                env=os.environ | {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
                stdout=f,
                stderr=f
            )
            
        elif name in ["ours", "base", "fsdc"]:
            nl = config['num_layers']
            if name == "base":
                cmd=["--always-use-gpu"]
                num_gpu_blocks_override = config["num_gpu_blocks_override"]
                swap_space = config["swap_space"] // 8
            elif name == "ours":
                cmd=["--extra-layer-for-cprf"]
                num_gpu_blocks_override = config["num_gpu_blocks_override"] * nl // (nl + 1)
                swap_space = config["swap_space"]
            else:
                cmd=["--disable-partial-offl", "--extra-layer-for-cprf"]
                num_gpu_blocks_override = config["num_gpu_blocks_override"] * nl // (nl + 1)
                swap_space = config["swap_space"]
            
            cmd = numacmd + [
                sys.executable, "-m", "swiftllm.server.api_server",
                "--port", "8000",
                "--model-path", config["model_path"],
                "--block-size", str(config["block_size"]),
                "--max-blocks-per-seq", str((config["max_num_batched_tokens"] - 1) // config["block_size"] + 1),
                "--max-seqs-in-block-table", str(config["max_num_seqs"]),
                "--max-batch-size", str(config["max_num_seqs"]),
                "--max-tokens-in-batch", str(config["max_num_batched_tokens"]),
                "--tensor-parallel-degree", str(config["tensor_parallel_size"]),
                # "--gpu-mem-utilization", str(config["gpu_memory_utilization"]),
                "--num-gpu-blocks-override", str(num_gpu_blocks_override),
                "--swap-space", str(swap_space),
                "--library-path", f"{neo_dir}/pacpu/build/{config['library']}",
                "--profile-result-path", f"{neo_dir}/profile_results/",
            ] + cmd

            server_proc = subprocess.Popen(
                cmd, 
                stdout=f,
                stderr=f
            )
            
        else:
            raise ValueError(f"Unknown server name: {name}")
        
        # Check the server log every 5s, until the starting keyword is found
        time_counter = 0
        while True:
            time.sleep(5)
            time_counter += 5
            logger.info(f"{time_counter}s elapsed, checking server log ...")
            with open(f"{cur_dir}/{name}-server.log", "r") as f:
                if "Started server process" in f.read():
                    break
        time.sleep(0.5)
        
        logger.info("Server started")


def stop_server():
    """
    Stop the server
    """
    assert server_proc is not None, "Server not started"
    server_proc.terminate()
    logger.info("Server stopped")

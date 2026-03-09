python3 client_prefix.py

python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000\
    --num-prompts 2000 \
    --random-output-len 30 \
    --num-families 50 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name custom_shared_prefix_dataset\
    --dataset-path /home/preeti/sglang/custom_shared_prefix_dataset.json \
    --fam-size 100 \ # this should be equal to batch size for homogeneous batching
    # --request-rate 50 \


# python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --num-prompts 10000 \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct --dataset-name random --random-input-len 30 --random-output-len 30 --request-rate 1000

# python3 client_prefix.py
# python3 client_os_req.py

# /home/preeti/sglang/mlfq_shared_prefix_dataset.json

# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py

# """
# Benchmark online serving with dynamic requests.

# Usage:
# python3 -m sglang.bench_serving --backend sglang --num-prompt 10

# python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
# """

# import argparse
# import asyncio
# import base64
# import io
# import json
# import os
# import pickle
# import random
# import resource
# import sys
# import time
# import traceback
# import warnings
# from argparse import ArgumentParser
# from dataclasses import dataclass, field
# from datetime import datetime
# from json import JSONDecodeError
# from pathlib import Path
# from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

# import aiohttp
# import numpy as np
# import requests
# from tqdm.asyncio import tqdm
# from transformers import (
#     AutoTokenizer,
#     PreTrainedTokenizer,
#     PreTrainedTokenizerBase,
#     PreTrainedTokenizerFast,
# )

# ASSISTANT_SUFFIX = "Assistant:"

# global args


# # don't want to import sglang package here
# def _get_bool_env_var(name: str, default: str = "false") -> bool:
#     value = os.getenv(name, default)
#     return value.lower() in ("true", "1")


# def _create_bench_client_session():
#     # When the pressure is big, the read buffer could be full before aio thread read
#     # the content. We increase the read_bufsize from 64K to 10M.
#     # Define constants for timeout and buffer size for clarity and maintainability
#     BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
#     BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB

#     aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
#     return aiohttp.ClientSession(
#         timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
#     )


# @dataclass
# class RequestFuncInput:
#     prompt: str
#     api_url: str
#     prompt_len: int
#     output_len: int
#     model: str
#     image_data: Optional[List[str]]
#     extra_request_body: Dict[str, Any]
#     timestamp: Optional[float] = None


# @dataclass
# class RequestFuncOutput:
#     generated_text: str = ""
#     success: bool = False
#     latency: float = 0.0
#     ttft: float = 0.0  # Time to first token
#     itl: List[float] = field(default_factory=list)  # List of inter-token latencies
#     prompt_len: int = 0
#     error: str = ""
#     output_len: int = 0

#     @staticmethod
#     def init_new(request_func_input: RequestFuncInput):
#         output = RequestFuncOutput()
#         output.prompt_len = request_func_input.prompt_len
#         return output


# def remove_prefix(text: str, prefix: str) -> str:
#     return text[len(prefix) :] if text.startswith(prefix) else text


# def remove_suffix(text: str, suffix: str) -> str:
#     return text[: -len(suffix)] if text.endswith(suffix) else text


# def get_auth_headers() -> Dict[str, str]:
#     api_key = os.environ.get("OPENAI_API_KEY")
#     if api_key:
#         return {"Authorization": f"Bearer {api_key}"}
#     else:
#         return {}

# async def async_request_truss(
#     request_func_input: RequestFuncInput,
#     pbar: Optional[tqdm] = None,
# ) -> RequestFuncOutput:
#     api_url = request_func_input.api_url

#     prompt = request_func_input.prompt

#     async with _create_bench_client_session() as session:
#         payload = {
#             "model": request_func_input.model,
#             "prompt": prompt,
#             "temperature": 0.0,
#             "best_of": 1,
#             "max_tokens": request_func_input.output_len,
#             "stream": not args.disable_stream,
#             "ignore_eos": not args.disable_ignore_eos,
#             **request_func_input.extra_request_body,
#         }
#         headers = get_auth_headers()

#         output = RequestFuncOutput.init_new(request_func_input)

#         generated_text = ""
#         ttft = 0.0
#         st = time.perf_counter()
#         most_recent_timestamp = st
#         try:
#             async with session.post(
#                 url=api_url, json=payload, headers=headers
#             ) as response:
#                 if response.status == 200:
#                     async for chunk_bytes in response.content:
#                         chunk_bytes = chunk_bytes.strip()
#                         if not chunk_bytes:
#                             continue

#                         chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
#                         latency = time.perf_counter() - st
#                         if chunk == "[DONE]":
#                             pass
#                         else:
#                             data = json.loads(chunk)

#                             # NOTE: Some completion API might have a last
#                             # usage summary response without a token so we
#                             # want to check a token was generated
#                             if data["choices"][0]["text"]:
#                                 timestamp = time.perf_counter()
#                                 # First token
#                                 if ttft == 0.0:
#                                     ttft = time.perf_counter() - st
#                                     output.ttft = ttft

#                                 # Decoding phase
#                                 else:
#                                     output.itl.append(timestamp - most_recent_timestamp)

#                                 most_recent_timestamp = timestamp
#                                 generated_text += data["choices"][0]["text"]

#                     output.generated_text = generated_text
#                     output.success = True
#                     output.latency = latency
#                     output.output_len = request_func_input.output_len
#                 else:
#                     output.error = response.reason or ""
#                     output.success = False
#         except Exception:
#             output.success = False
#             exc_info = sys.exc_info()
#             output.error = "".join(traceback.format_exception(*exc_info))

#     if pbar:
#         pbar.update(1)
#     return output


# async def async_request_sglang_generate(
#     request_func_input: RequestFuncInput,
#     pbar: Optional[tqdm] = None,
# ) -> RequestFuncOutput:
#     api_url = request_func_input.api_url
#     prompt = request_func_input.prompt

#     async with _create_bench_client_session() as session:
#         payload = {
#             ("text" if isinstance(prompt, str) else "input_ids"): prompt,
#             "sampling_params": {
#                 "temperature": 0.0,
#                 "max_new_tokens": request_func_input.output_len,
#                 "ignore_eos": not args.disable_ignore_eos,
#             },
#             "stream": not args.disable_stream,
#             "return_logprob": args.return_logprob,
#             "logprob_start_len": -1,
#             **request_func_input.extra_request_body,
#         }

#         # Add image data if available (list of image urls/base64)
#         if request_func_input.image_data:
#             payload["image_data"] = request_func_input.image_data

#         headers = get_auth_headers()

#         output = RequestFuncOutput.init_new(request_func_input)

#         generated_text = ""
#         output_len = request_func_input.output_len
#         ttft = 0.0
#         st = time.perf_counter()
#         most_recent_timestamp = st
#         last_output_len = 0
#         try:
#             async with session.post(
#                 url=api_url, json=payload, headers=headers
#             ) as response:
#                 if response.status == 200:
#                     async for chunk_bytes in response.content:
#                         chunk_bytes = chunk_bytes.strip()
#                         if not chunk_bytes:
#                             continue

#                         chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
#                         latency = time.perf_counter() - st
#                         if chunk == "[DONE]":
#                             pass
#                         else:
#                             data = json.loads(chunk)

#                             # NOTE: Some completion API might have a last
#                             # usage summary response without a token so we
#                             # want to check a token was generated
#                             if "text" in data and data["text"]:
#                                 timestamp = time.perf_counter()
#                                 generated_text = data["text"]
#                                 output_len = data["meta_info"]["completion_tokens"]

#                                 # First token
#                                 if ttft == 0.0:
#                                     ttft = time.perf_counter() - st
#                                     output.ttft = ttft

#                                 # Decoding phase
#                                 else:
#                                     num_new_tokens = output_len - last_output_len
#                                     if num_new_tokens == 0:
#                                         continue
#                                     adjust_itl = (
#                                         timestamp - most_recent_timestamp
#                                     ) / num_new_tokens
#                                     output.itl.extend([adjust_itl] * num_new_tokens)

#                                 most_recent_timestamp = timestamp
#                                 last_output_len = output_len

#                     output.generated_text = generated_text
#                     output.success = True
#                     output.latency = latency
#                     output.output_len = output_len
#                 else:
#                     output.error = response.reason or ""
#                     output.success = False
#         except Exception:
#             output.success = False
#             exc_info = sys.exc_info()
#             output.error = "".join(traceback.format_exception(*exc_info))
#             print(f"{output.error=}")

#     if pbar:
#         pbar.update(1)
#     return output

# def get_model(pretrained_model_name_or_path: str) -> str:
#     if os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() == "true":
#         import huggingface_hub.constants
#         from modelscope import snapshot_download

#         model_path = snapshot_download(
#             model_id=pretrained_model_name_or_path,
#             local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
#             ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
#         )

#         return model_path
#     return pretrained_model_name_or_path


# def get_tokenizer(
#     pretrained_model_name_or_path: str,
# ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
#     assert (
#         pretrained_model_name_or_path is not None
#         and pretrained_model_name_or_path != ""
#     )
#     if pretrained_model_name_or_path.endswith(
#         ".json"
#     ) or pretrained_model_name_or_path.endswith(".model"):
#         from sglang.srt.hf_transformers_utils import get_tokenizer

#         return get_tokenizer(pretrained_model_name_or_path)

#     if pretrained_model_name_or_path is not None and not os.path.exists(
#         pretrained_model_name_or_path
#     ):
#         pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
#     return AutoTokenizer.from_pretrained(
#         pretrained_model_name_or_path, trust_remote_code=True
#     )


# def get_dataset(args, tokenizer):
#     tokenize_prompt = getattr(args, "tokenize_prompt", False)
#     if args.dataset_name == "sharegpt":
#         assert not tokenize_prompt
#         input_requests = sample_sharegpt_requests(
#             dataset_path=args.dataset_path,
#             num_requests=args.num_prompts,
#             tokenizer=tokenizer,
#             fixed_output_len=args.sharegpt_output_len,
#             context_len=args.sharegpt_context_len,
#             prompt_suffix=args.prompt_suffix,
#             apply_chat_template=args.apply_chat_template,
#         )
#     elif args.dataset_name == "generated-shared-prefix":
#         assert not tokenize_prompt
#         input_requests = sample_generated_shared_prefix_requests(
#             num_groups=args.gsp_num_groups,
#             prompts_per_group=args.gsp_prompts_per_group,
#             system_prompt_len=args.gsp_system_prompt_len,
#             question_len=args.gsp_question_len,
#             output_len=args.gsp_output_len,
#             tokenizer=tokenizer,
#             args=args,
#         )
#     else:
#         raise ValueError(f"Unknown dataset: {args.dataset_name}")
#     return input_requests


# ASYNC_REQUEST_FUNCS = {
#     "sglang": async_request_sglang_generate,
#     "sglang-native": async_request_sglang_generate,
#     "sglang-oai": async_request_openai_completions,
#     "sglang-oai-chat": async_request_openai_chat_completions,
#     "lmdeploy": async_request_openai_completions,
#     "lmdeploy-chat": async_request_openai_chat_completions,
#     "trt": async_request_trt_llm,
#     "gserver": async_request_gserver,
#     "truss": async_request_truss,
# }


# @dataclass
# class BenchmarkMetrics:
#     completed: int
#     total_input: int
#     total_output: int
#     total_output_retokenized: int
#     request_throughput: float
#     input_throughput: float
#     output_throughput: float
#     output_throughput_retokenized: float
#     total_throughput: float
#     total_throughput_retokenized: float
#     mean_ttft_ms: float
#     median_ttft_ms: float
#     std_ttft_ms: float
#     p99_ttft_ms: float
#     mean_tpot_ms: float
#     median_tpot_ms: float
#     std_tpot_ms: float
#     p99_tpot_ms: float
#     mean_itl_ms: float
#     median_itl_ms: float
#     std_itl_ms: float
#     p95_itl_ms: float
#     p99_itl_ms: float
#     max_itl_ms: float
#     mean_e2e_latency_ms: float
#     median_e2e_latency_ms: float
#     std_e2e_latency_ms: float
#     p99_e2e_latency_ms: float

# SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


# def download_and_cache_file(url: str, filename: Optional[str] = None):
#     """Read and cache a file from a url."""
#     if filename is None:
#         filename = os.path.join("/tmp", url.split("/")[-1])

#     # Check if the cache file already exists
#     if is_file_valid_json(filename):
#         return filename

#     print(f"Downloading from {url} to {filename}")

#     # Stream the response to show the progress bar
#     response = requests.get(url, stream=True)
#     response.raise_for_status()  # Check for request errors

#     # Total size of the file in bytes
#     total_size = int(response.headers.get("content-length", 0))
#     chunk_size = 1024  # Download in chunks of 1KB

#     # Use tqdm to display the progress bar
#     with open(filename, "wb") as f, tqdm(
#         desc=filename,
#         total=total_size,
#         unit="B",
#         unit_scale=True,
#         unit_divisor=1024,
#     ) as bar:
#         for chunk in response.iter_content(chunk_size=chunk_size):
#             f.write(chunk)
#             bar.update(len(chunk))

#     return filename


# def is_file_valid_json(path):
#     if not os.path.isfile(path):
#         return False

#     # TODO can fuse into the real file open later
#     try:
#         with open(path) as f:
#             json.load(f)
#         return True
#     except JSONDecodeError as e:
#         print(
#             f"{path} exists but json loading fails ({e=}), thus treat as invalid file"
#         )
#         return False


# @dataclass
# class DatasetRow:
#     prompt: str
#     prompt_len: int
#     output_len: int
#     image_data: Optional[List[str]] = None
#     timestamp: Optional[float] = None

# def sample_sharegpt_requests(
#     dataset_path: str,
#     num_requests: int,
#     tokenizer: PreTrainedTokenizerBase,
#     fixed_output_len: Optional[int] = None,
#     context_len: Optional[int] = None,
#     prompt_suffix: Optional[str] = "",
#     apply_chat_template=False,
# ) -> List[DatasetRow]:
#     if fixed_output_len is not None and fixed_output_len < 4:
#         raise ValueError("output_len too small")

#     # Download sharegpt if necessary
#     if not is_file_valid_json(dataset_path) and dataset_path == "":
#         dataset_path = download_and_cache_file(SHAREGPT_URL)

#     # Load the dataset.
#     with open(dataset_path) as f:
#         dataset = json.load(f)

#     # Filter out the conversations with less than 2 turns.
#     dataset = [
#         data
#         for data in dataset
#         if len(data.get("conversations", data.get("conversation", []))) >= 2
#     ]
#     # Only keep the first two turns of each conversation.
#     dataset = [
#         (
#             data.get("conversations", data.get("conversation", []))[0]["value"],
#             data.get("conversations", data.get("conversation", []))[1]["value"],
#         )
#         for data in dataset
#     ]

#     # Shuffle the dataset.
#     random.shuffle(dataset)

#     # Filter out sequences that are too long or too short
#     filtered_dataset: List[DatasetRow] = []
#     for i in range(len(dataset)):
#         if len(filtered_dataset) == num_requests:
#             break

#         # Tokenize the prompts and completions.
#         prompt = dataset[i][0]
#         if prompt_suffix:
#             prompt = (
#                 remove_suffix(prompt, ASSISTANT_SUFFIX)
#                 + prompt_suffix
#                 + ASSISTANT_SUFFIX
#             )

#         if apply_chat_template:
#             prompt = tokenizer.apply_chat_template(
#                 [{"role": "user", "content": prompt}],
#                 add_generation_prompt=True,
#                 tokenize=False,
#             )
#             prompt = prompt.replace(tokenizer.bos_token, "")

#         prompt_token_ids = tokenizer.encode(prompt)
#         completion = dataset[i][1]
#         completion_token_ids = tokenizer.encode(completion)
#         prompt_len = len(prompt_token_ids)
#         output_len = (
#             len(completion_token_ids) if fixed_output_len is None else fixed_output_len
#         )

#         if prompt_len < 2 or output_len < 2:
#             # Prune too short sequences.
#             continue

#         if context_len and prompt_len + output_len > context_len:
#             # Prune too long sequences.
#             continue

#         filtered_dataset.append(
#             DatasetRow(prompt=prompt, prompt_len=prompt_len, output_len=output_len)
#         )

#     print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
#     print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
#     return filtered_dataset

# def gen_prompt(tokenizer, token_num):
#     """Generate a random prompt of specified token length using tokenizer vocabulary."""
#     all_available_tokens = list(tokenizer.get_vocab().values())
#     selected_tokens = random.choices(all_available_tokens, k=token_num)
#     return tokenizer.decode(selected_tokens)


# def get_gen_prefix_cache_path(args, tokenizer):
#     """Create cache directory under ~/.cache/sglang/benchmark"""
#     cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

#     # Create a unique cache filename based on the generation parameters
#     cache_key = (
#         f"gen_shared_prefix_{args.gsp_num_groups}_{args.gsp_prompts_per_group}_"
#         f"{args.gsp_system_prompt_len}_{args.gsp_question_len}_{args.gsp_output_len}_"
#         f"{tokenizer.__class__.__name__}.pkl"
#     )
#     return cache_dir / cache_key


# def sample_generated_shared_prefix_requests(
#     num_groups: int,
#     prompts_per_group: int,
#     system_prompt_len: int,
#     question_len: int,
#     output_len: int,
#     tokenizer: PreTrainedTokenizerBase,
#     args: argparse.Namespace,
# ) -> List[DatasetRow]:
#     """Generate benchmark requests with shared system prompts using random tokens and caching."""
#     cache_path = get_gen_prefix_cache_path(args, tokenizer)

#     # Try to load from cache first
#     if cache_path.exists():
#         print(f"\nLoading cached generated input data from {cache_path}")
#         with open(cache_path, "rb") as f:
#             return pickle.load(f)

#     print("\nGenerating new input data...")

#     # Generate system prompts for each group
#     system_prompts = []
#     for _ in range(num_groups):
#         system_prompt = gen_prompt(tokenizer, system_prompt_len)
#         system_prompts.append(system_prompt)

#     # Generate questions
#     questions = []
#     for _ in range(num_groups * prompts_per_group):
#         question = gen_prompt(tokenizer, question_len)
#         questions.append(question)

#     # Combine system prompts with questions
#     input_requests = []
#     total_input_tokens = 0
#     total_output_tokens = 0

#     for group_idx in tqdm(range(num_groups), desc="Generating system prompt"):
#         system_prompt = system_prompts[group_idx]
#         for prompt_idx in tqdm(
#             range(prompts_per_group), desc="Generating questions", leave=False
#         ):
#             question = questions[group_idx * prompts_per_group + prompt_idx]
#             full_prompt = f"{system_prompt}\n\n{question}"
#             prompt_len = len(tokenizer.encode(full_prompt))

#             input_requests.append(
#                 DatasetRow(
#                     prompt=full_prompt, prompt_len=prompt_len, output_len=output_len
#                 )
#             )
#             total_input_tokens += prompt_len
#             total_output_tokens += output_len

#     # Shuffle questions
#     random.shuffle(input_requests)

#     # Print statistics
#     print(f"\nGenerated shared prefix dataset statistics:")
#     print(f"Number of groups: {num_groups}")
#     print(f"Prompts per group: {prompts_per_group}")
#     print(f"Total prompts: {len(input_requests)}")
#     print(f"Total input tokens: {total_input_tokens}")
#     print(f"Total output tokens: {total_output_tokens}")
#     print(
#         f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
#     )
#     print(
#         f"Average question length: {sum(len(tokenizer.encode(q)) for q in questions) / len(questions):.1f} tokens\n"
#     )

#     # Save to cache
#     cache_path.parent.mkdir(parents=True, exist_ok=True)
#     print(f"Caching generated input data to {cache_path}")
#     with open(cache_path, "wb") as f:
#         pickle.dump(input_requests, f)

#     return input_requests


# async def get_request(
#     input_requests: List[DatasetRow],
#     request_rate: float,
#     slowdown_factor: float = 1.0,
# ) -> AsyncGenerator[DatasetRow, None]:
#     input_requests_iter = iter(input_requests)
#     for request in input_requests_iter:
#         yield request

#         if request_rate == float("inf"):
#             # If the request rate is infinity, then we don't need to wait.
#             continue

#         # Sample the request interval from the exponential distribution.
#         interval = np.random.exponential(1.0 / request_rate)
#         # The next request will be sent after the interval.
#         await asyncio.sleep(interval)

# async def benchmark(
#     backend: str,
#     api_url: str,
#     base_url: str,
#     model_id: str,
#     tokenizer: PreTrainedTokenizerBase,
#     input_requests: List[DatasetRow],
#     request_rate: float,
#     disable_tqdm: bool,
#     extra_request_body: Dict[str, Any],
#     profile: bool,
#     pd_separated: bool = False,
#     warmup_requests: int = 1,
# ):
#     if backend in ASYNC_REQUEST_FUNCS:
#         request_func = ASYNC_REQUEST_FUNCS[backend]
#     else:
#         raise ValueError(f"Unknown backend: {backend}")

#     async def limited_request_func(request_func_input, pbar):
#         if semaphore is None:
#             return await request_func(request_func_input=request_func_input, pbar=pbar)
#         async with semaphore:
#             return await request_func(request_func_input=request_func_input, pbar=pbar)

#     # Warmup
#     print(f"Starting warmup with {warmup_requests} sequences...")

#     test_request = input_requests[0]

#     # Create the test input once
#     test_input = RequestFuncInput(
#         model=model_id,
#         prompt=test_request.prompt,
#         api_url=api_url,
#         prompt_len=test_request.prompt_len,
#         output_len=min(test_request.output_len, 32),
#         image_data=test_request.image_data,
#         extra_request_body=extra_request_body,
#     )

#     # Run warmup requests
#     warmup_tasks = []
#     for _ in range(warmup_requests):
#         warmup_tasks.append(
#             asyncio.create_task(request_func(request_func_input=test_input))
#         )

#     warmup_outputs = await asyncio.gather(*warmup_tasks)

#     # Check if at least one warmup request succeeded
#     if warmup_requests > 0 and not any(output.success for output in warmup_outputs):
#         raise ValueError(
#             "Warmup failed - Please make sure benchmark arguments "
#             f"are correctly specified. Error: {warmup_outputs[0].error}"
#         )
#     else:
#         print(
#             f"Warmup completed with {args.warmup_requests} sequences. Starting main benchmark run..."
#         )

#     time.sleep(1.0)

#     # Start profiler
#     if profile:
#         print("Starting profiler...")
#         profile_output = await async_request_profile(
#             api_url=base_url + "/start_profile"
#         )
#         if profile_output.success:
#             print("Profiler started")

#     # Run all requests
#     benchmark_start_time = time.perf_counter()
#     tasks: List[asyncio.Task] = []
#     pbar_total = len(input_requests)
    
#     request_generator = get_request(input_requests, request_rate)

#     pbar = None if disable_tqdm else tqdm(total=pbar_total)
#     async for request in request_generator:

#         request_func_input = RequestFuncInput(
#             model=model_id,
#             prompt=request.prompt,
#             api_url=api_url,
#             prompt_len=request.prompt_len,
#             output_len=request.output_len,
#             image_data=request.image_data,
#             extra_request_body=extra_request_body,
#             timestamp=request.timestamp,
#         )

#         tasks.append(
#             asyncio.create_task(
#                 limited_request_func(request_func_input=request_func_input, pbar=pbar)
#             )
#         )
#     outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

#     # Stop profiler
#     if profile:
#         print("Stopping profiler...")
#         profile_output = await async_request_profile(api_url=base_url + "/stop_profile")
#         if profile_output.success:
#             print("Profiler stopped")

#     if pbar is not None:
#         pbar.close()

#     if "sglang" in backend:
#         server_info = requests.get(base_url + "/get_server_info")
#         if server_info.status_code == 200:
#             server_info_json = server_info.json()
#             if "decode" in server_info_json:
#                 server_info_json = server_info_json["decode"][0]
#             accept_length = server_info_json["internal_states"][0].get(
#                 "avg_spec_accept_length", None
#             )
#         else:
#             accept_length = None
#     else:
#         accept_length = None

#     # Compute metrics and print results
#     benchmark_duration = time.perf_counter() - benchmark_start_time
#     metrics, output_lens = calculate_metrics(
#         input_requests=input_requests,
#         outputs=outputs,
#         dur_s=benchmark_duration,
#         tokenizer=tokenizer,
#         backend=backend,
#     )

#     if (
#         metrics.median_ttft_ms is not None
#         and metrics.mean_itl_ms is not None
#         and metrics.output_throughput is not None
#     ):
#         result = {
#             # Arguments
#             "backend": args.backend,
#             "dataset_name": args.dataset_name,
#             "request_rate": "trace" if use_trace_timestamps else request_rate,
#             "sharegpt_output_len": args.sharegpt_output_len,
#             # Results
#             "duration": benchmark_duration,
#             "completed": metrics.completed,
#             "total_input_tokens": metrics.total_input,
#             "total_output_tokens": metrics.total_output,
#             "total_output_tokens_retokenized": metrics.total_output_retokenized,
#             "request_throughput": metrics.request_throughput,
#             "input_throughput": metrics.input_throughput,
#             "output_throughput": metrics.output_throughput,
#             "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
#             "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
#             "std_e2e_latency_ms": metrics.std_e2e_latency_ms,
#             "p99_e2e_latency_ms": metrics.p99_e2e_latency_ms,
#             "mean_ttft_ms": metrics.mean_ttft_ms,
#             "median_ttft_ms": metrics.median_ttft_ms,
#             "std_ttft_ms": metrics.std_ttft_ms,
#             "p99_ttft_ms": metrics.p99_ttft_ms,
#             "mean_tpot_ms": metrics.mean_tpot_ms,
#             "median_tpot_ms": metrics.median_tpot_ms,
#             "std_tpot_ms": metrics.std_tpot_ms,
#             "p99_tpot_ms": metrics.p99_tpot_ms,
#             "mean_itl_ms": metrics.mean_itl_ms,
#             "median_itl_ms": metrics.median_itl_ms,
#             "std_itl_ms": metrics.std_itl_ms,
#             "p95_itl_ms": metrics.p95_itl_ms,
#             "p99_itl_ms": metrics.p99_itl_ms,
#             "accept_length": accept_length,
#         }
#     else:
#         print(f"Error running benchmark for request rate: {request_rate}")
#         print("-" * 30)

#     # Determine output file name
#     if args.output_file:
#         output_file_name = args.output_file
#     else:
#         now = datetime.now().strftime("%m%d")
#         output_file_name = (
#             f"{args.backend}_{now}_{args.num_prompts}_{args.dataset_name}.jsonl"
#         )

#     result_details = {
#         "input_lens": [output.prompt_len for output in outputs],
#         "output_lens": output_lens,
#         "ttfts": [output.ttft for output in outputs],
#         "itls": [output.itl for output in outputs],
#         "generated_texts": [output.generated_text for output in outputs],
#         "errors": [output.error for output in outputs],
#     }

#     # Append results to a JSONL file
#     with open(output_file_name, "a") as file:
#         if args.output_details:
#             result_for_dump = result | result_details
#         else:
#             result_for_dump = result
#         file.write(json.dumps(result_for_dump) + "\n")

#     return result | result_details

# def run_benchmark(args_: argparse.Namespace):
#     global args
#     args = args_

#     # Set default value for warmup_requests if not present
#     if not hasattr(args, "warmup_requests"):
#         args.warmup_requests = 1

#     if not hasattr(args, "output_details"):
#         args.output_details = False

#     if not hasattr(args, "tokenize_prompt"):
#         args.tokenize_prompt = False

#     print(f"benchmark_args={args}")

#     # Set global environments
#     set_ulimit()
#     random.seed(args.seed)
#     np.random.seed(args.seed)

#     extra_request_body = {}
#     if args.extra_request_body:
#         extra_request_body = json.loads(args.extra_request_body)

#     if args.tokenize_prompt:
#         assert (
#             args.backend == "sglang"
#         ), "`--tokenize-prompt` only compatible with `--backend sglang` currently"

#     # Set url
#     if args.port is None:
#         args.port = {
#             "sglang": 30000,
#             "sglang-native": 30000,
#             "sglang-oai": 30000,
#             "lmdeploy": 23333,
#             "trt": 8000,
#             "gserver": 9988,
#             "truss": 8080,
#         }.get(args.backend, 30000)

#     model_url = (
#         f"{args.base_url}/v1/models"
#         if args.base_url
#         else f"http://{args.host}:{args.port}/v1/models"
#     )

#     if args.backend in ["sglang", "sglang-native"]:
#         api_url = (
#             f"{args.base_url}/generate"
#             if args.base_url
#             else f"http://{args.host}:{args.port}/generate"
#         )
#     elif args.backend == "trt":
#         api_url = (
#             f"{args.base_url}/v2/models/ensemble/generate_stream"
#             if args.base_url
#             else f"http://{args.host}:{args.port}/v2/models/ensemble/generate_stream"
#         )
#         if args.model is None:
#             print("Please provide a model using `--model` when using `trt` backend.")
#             sys.exit(1)
#     elif args.backend == "gserver":
#         api_url = args.base_url if args.base_url else f"{args.host}:{args.port}"
#         args.model = args.model or "default"
#     elif args.backend == "truss":
#         api_url = (
#             f"{args.base_url}/v1/models/model:predict"
#             if args.base_url
#             else f"http://{args.host}:{args.port}/v1/models/model:predict"
#         )
#     base_url = (
#         f"http://{args.host}:{args.port}" if args.base_url is None else args.base_url
#     )

#     # Get model name
#     if args.model is None:
#         if args.backend == "truss":
#             print(
#                 "Please provide a model with `--model` when using truss backend. e.g. --model meta-llama/Llama-3.1-8B-Instruct"
#             )
#             sys.exit(1)
#         try:
#             response = requests.get(model_url, headers=get_auth_headers())
#             model_list = response.json().get("data", [])
#             args.model = model_list[0]["id"] if model_list else None
#         except Exception as e:
#             print(f"Failed to fetch model from {model_url}. Error: {e}")
#             print(
#                 "Please specify the correct host and port using `--host` and `--port`."
#             )
#             sys.exit(1)

#     if args.model is None:
#         print("No model specified or found. Please provide a model using `--model`.")
#         sys.exit(1)

#     if not check_chat_template(args.model):
#         print(
#             "\nWARNING It is recommended to use the `Chat` or `Instruct` model for benchmarking.\n"
#             "Because when the tokenizer counts the output tokens, if there is gibberish, it might count incorrectly.\n"
#         )

#     print(f"{args}\n")

#     # Read dataset
#     backend = args.backend
#     model_id = args.model
#     tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
#     tokenizer = get_tokenizer(tokenizer_id)
#     input_requests = get_dataset(args, tokenizer)

#     return asyncio.run(
#         benchmark(
#             backend=backend,
#             api_url=api_url,
#             base_url=base_url,
#             model_id=model_id,
#             tokenizer=tokenizer,
#             input_requests=input_requests,
#             request_rate=args.request_rate,
#             disable_tqdm=args.disable_tqdm,
#             extra_request_body=extra_request_body,
#             profile=args.profile,
#             pd_separated=args.pd_separated,
#             warmup_requests=args.warmup_requests,
#         )
#     )


# if __name__ == "__main__":
#     parser = ArgumentParser(description="Benchmark the online serving throughput.")
#     parser.add_argument(
#         "--backend",
#         type=str,
#         choices=list(ASYNC_REQUEST_FUNCS.keys()),
#         default="sglang",
#         help="Must specify a backend, depending on the LLM Inference Engine.",
#     )
#     parser.add_argument(
#         "--base-url",
#         type=str,
#         default=None,
#         help="Server or API base url if not using http host and port.",
#     )
#     parser.add_argument(
#         "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
#     )
#     parser.add_argument(
#         "--port",
#         type=int,
#         help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
#     )
#     parser.add_argument(
#         "--dataset-name",
#         type=str,
#         default="sharegpt",
#         choices=[
#             "sharegpt",
#             "generated-shared-prefix",
#         ],
#         help="Name of the dataset to benchmark on.",
#     )
#     parser.add_argument(
#         "--dataset-path", type=str, default="", help="Path to the dataset."
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
#     )
#     parser.add_argument(
#         "--tokenizer",
#         type=str,
#         help="Name or path of the tokenizer. If not set, using the model conf.",
#     )
#     parser.add_argument(
#         "--num-prompts",
#         type=int,
#         default=1000,
#         help="Number of prompts to process. Default is 1000.",
#     )
#     parser.add_argument(
#         "--sharegpt-output-len",
#         type=int,
#         default=None,
#         help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
#     )
#     parser.add_argument(
#         "--sharegpt-context-len",
#         type=int,
#         default=None,
#         help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
#     )
#     parser.add_argument(
#         "--request-rate",
#         type=float,
#         default=float("inf"),
#         help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
#         "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
#     )
#     parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
#     parser.add_argument(
#         "--output-details", action="store_true", help="Output details of benchmarking."
#     )
#     parser.add_argument("--seed", type=int, default=1, help="The random seed.")
#     parser.add_argument(
#         "--disable-ignore-eos",
#         action="store_true",
#         help="Disable ignoring EOS.",
#     )
#     parser.add_argument(
#         "--extra-request-body",
#         metavar='{"key1": "value1", "key2": "value2"}',
#         type=str,
#         help="Append given JSON object to the request payload. You can use this to specify"
#         "additional generate params like sampling params.",
#     )
#     parser.add_argument(
#         "--profile",
#         action="store_true",
#         help="Use Torch Profiler. The endpoint must be launched with "
#         "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
#     )
#     parser.add_argument(
#         "--prompt-suffix",
#         type=str,
#         default="",
#         help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
#     )
#     parser.add_argument(
#         "--warmup-requests",
#         type=int,
#         default=1,
#         help="Number of warmup requests to run before the benchmark",
#     )
#     parser.add_argument(
#         "--tokenize-prompt",
#         action="store_true",
#         help="Use integer ids instead of string for inputs. Useful to control prompt lengths accurately",
#     )

#     group = parser.add_argument_group("generated-shared-prefix dataset arguments")
#     group.add_argument(
#         "--gsp-num-groups",
#         type=int,
#         default=64,
#         help="Number of system prompt groups for generated-shared-prefix dataset",
#     )
#     group.add_argument(
#         "--gsp-prompts-per-group",
#         type=int,
#         default=16,
#         help="Number of prompts per system prompt group for generated-shared-prefix dataset",
#     )
#     group.add_argument(
#         "--gsp-system-prompt-len",
#         type=int,
#         default=2048,
#         help="Target length in tokens for system prompts in generated-shared-prefix dataset",
#     )
#     group.add_argument(
#         "--gsp-question-len",
#         type=int,
#         default=128,
#         help="Target length in tokens for questions in generated-shared-prefix dataset",
#     )
#     group.add_argument(
#         "--gsp-output-len",
#         type=int,
#         default=256,
#         help="Target length in tokens for outputs in generated-shared-prefix dataset",
#     )    
#     args = parser.parse_args()
#     run_benchmark(args)

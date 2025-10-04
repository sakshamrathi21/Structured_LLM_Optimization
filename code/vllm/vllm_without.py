from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from system_prompt import system_prompt, queries
from gpu_mem import stop_logging, start_logging
import torch

def main():
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name = "qwen/qwen2.5-0.5b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    system_prompt_halved = system_prompt[0:len(system_prompt)]
    prompts = []
    for q in queries:
        chat_messages = [
            {"role": "system", "content": system_prompt_halved},
            {"role": "user", "content": q},
        ]
        full_prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(full_prompt)
    enable_chunked_prefill = True
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.75,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=16384,
        enable_prefix_caching=True,
        max_num_seqs=750,
    )
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # meta-llama/Llama-3.1-8B-Instruct
    prompts *= 200
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    llm.generate(prompts[0:10], sampling_params)
    return
    for interval in [20000]:
        initial_start_time = time.time()
        multiplier = int(interval/10)
        if multiplier == 0:
            multiplier += 1
        multiplier = 1
        all_prompts = prompts * multiplier
        for i in range(0, len(all_prompts), interval):
            start_time = time.time()
            outputs = llm.generate(all_prompts[i:i+interval], sampling_params)
            # time.sleep(max(0, 1 - time.time() + start_time))
        print(f"Throughput for interval ({interval}): ", len(all_prompts) / (time.time() - initial_start_time), "req/s (enabling chunked prefill:", enable_chunked_prefill, ")")

    # prompts *= 100
    # start_time = time.time()
    # outputs = llm.generate(prompts, sampling_params)
    # end_time = time.time()
    # print("Throughput: ", len(outputs) / (end_time - start_time), "req/s (enabling chunked prefill:", enable_chunked_prefill, ")")
    
    # for interval in [1/5, 1/10, 1/20, 1/30, 1/50, 1/100]:
    # # prompts = prompts[0:10]
    # # for interval in [1/100]:
    #     start = time.time()
    #     outputs = llm._run_engine(
    #         use_tqdm=True,
    #         prompts=prompts,
    #         sampling_params=sampling_params,
    #         interval=interval
    #     )
    #     end = time.time()
    #     print("Throughput: ", len(outputs) / (end - start), "req/s (enabling chunked prefill:", enable_chunked_prefill, ")", "Interval:", interval, " Len outputs:", len(outputs))

    # for interval in [1/5, 1/10, 1/20, 1/30, 1/50, 1/100]:
    # all_prompts = prompts * 10
    # for interval in [100]:
    #     start = time.time()
    #     outputs = llm.generate2(
    #         use_tqdm=True,
    #         prompts=all_prompts,
    #         sampling_params=sampling_params,
    #         request_interval=interval,
    #         continuous=True,
    #     )
    #     end = time.time()
    #     print("Throughput: ", len(outputs) / (end - start), "req/s (enabling chunked prefill:", enable_chunked_prefill, ")", "Interval:", interval, " Len outputs:", len(outputs))
    #     time.sleep(2)
    # print("Time:", end - start, "s, Outputs:", len(outputs))
    # for q, out in zip(queries, outputs):
    #     print(f"\nUser: {q}\nAssistant: {out.outputs[0].text.strip()}")

if __name__ == "__main__":
    main()

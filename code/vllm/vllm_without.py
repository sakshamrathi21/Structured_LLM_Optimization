from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import cupy
from nvtx import annotate
from system_prompt import system_prompt, queries
from gpu_mem import stop_logging, start_logging

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.85,
        # enable_chunked_prefill=True,
    )
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # meta-llama/Llama-3.1-8B-Instruct
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    # prompts = prompts[:10]
    start = time.time()
    print("Generation Start Time:", start)
    # outputs = llm.generate(prompts, sampling_params)
    with annotate("LLM forward"):
        outputs = llm._run_engine(
            use_tqdm=True,
            prompts=prompts,
            sampling_params=sampling_params,
            interval=0.01
        )
    end = time.time()
    print("Generation End Time:", end)

    print("Time:", end - start, "s, Outputs:", len(outputs))
    # for q, out in zip(queries, outputs):
    #     print(f"\nUser: {q}\nAssistant: {out.outputs[0].text.strip()}")

if __name__ == "__main__":
    main()

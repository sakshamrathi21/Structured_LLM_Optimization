# sglang_offline_fixed.py
import asyncio
import os
import sys

# Optional: silence CUDA arch warning (set to the arches you need).
# os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5;8.6"   # example; set to your GPU archs if desired

def run():
    # import heavy libs inside main to avoid executing them during module import
    import sglang as sgl
    from sglang.utils import async_stream_and_merge, stream_and_merge

    # If you were passing torch_dtype anywhere, replace with dtype (deprecation)
    # e.g., sgl.Engine(..., dtype=torch.float16) instead of torch_dtype=...

    # create the engine here (inside main/guard)
    llm = sgl.Engine(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France’s capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]
    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    async def main_async():
        outputs = await llm.async_generate(prompts, sampling_params)
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print(f"Generated text: {output['text']}")

    asyncio.run(main_async())


if __name__ == "__main__":
    # Windows: necessary when using spawn; safe on other platforms
    import multiprocessing
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass

    # Optionally force 'fork' on POSIX to avoid spawn semantics (not recommended on macOS GUI)
    # try:
    #     multiprocessing.set_start_method('fork', force=True)
    # except RuntimeError:
    #     # start method already set or not available
    #     pass

    run()

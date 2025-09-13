import asyncio

def run():
    import sglang as sgl
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
    import multiprocessing
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass
    run()

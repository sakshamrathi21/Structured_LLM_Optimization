import os
import json
import asyncio
from fastapi import HTTPException
import aiohttp
from transformers import AutoTokenizer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

async def request_completions(
    api_url: str,
    prompt: str | list[int],
    output_len: int,
    model_path: str
):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:        
        payload = {
            "model": model_path,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0.0,
            "ignore_eos": True
        }

        async with session.post(url=api_url, json=payload) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=await response.text())
            data = json.loads(await response.text())

    return data['choices'][0]['text']


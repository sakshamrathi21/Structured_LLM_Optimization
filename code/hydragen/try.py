from hydragen.llama import HydragenLlamaForCausalLM
from transformers import AutoTokenizer
import torch

dtype = "auto"
device = "cuda"
pretrained_name = "meta-llama/Llama-3.1-8B-Instruct"

model = HydragenLlamaForCausalLM.from_pretrained(
    pretrained_name, torch_dtype=dtype, device_map=device
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id

base_prompt = "Harry Potter is"

base_prompt_inputs = tokenizer(base_prompt, return_tensors="pt")
base_ids = base_prompt_inputs.input_ids.to(device)

torch.manual_seed(0)

model.setup_caches(
    max_unique_batch_size=8, 
    max_unique_seq_length=16,
    max_shared_batch_sizes=[base_ids.shape[0]],
    max_shared_seq_lengths=[base_ids.shape[1]]
)

new_ids = model.generate(
    input_ids=base_ids,
    num_return_sequences=8,
    max_new_tokens=16,
    temperature=0.6,
)

print("\n\n\n".join(tokenizer.batch_decode(new_ids)))
import torch

def lsgpu(device_id=0):
    prop = torch.cuda.get_device_properties(device_id)
    print(f"GPU: {prop.name}")
    print(f"  Total memory: {prop.total_memory // 1024**2} MB")
    print(f"  Multiprocessors: {prop.multi_processor_count}")
    print(f"  Warp size: {prop.warp_size}")
    print(f"  Shared memory per block: {prop.shared_memory_per_block} bytes")
    print(f"  L2 cache size: {prop.L2_cache_size} bytes")
    print(f"  Max threads per SM: {prop.max_threads_per_multi_processor}")

lsgpu()

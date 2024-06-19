import argparse
import time
from distserve import OfflineLLM
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
import torch
import threading
from cuda import cuda

parser = argparse.ArgumentParser()
model = '/users/xyx/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/'
parser.add_argument('--model', type=str, help='The model to use', default=model)
args = parser.parse_args()


def print_current_ctx(i: int):
    err, ctx = cuda.cuCtxGetCurrent()
    print(f"{i}, {ctx}")

# Let's start Pytorch first
torch.cuda.init()

def thread(i: int):

    print_current_ctx(i)
    torch.zeros((1, 1), device='cuda')
    print_current_ctx(i)

    # Change the CUDA context afterwards?
    err, device = cuda.cuDeviceGet(0)
    err, cuda_context = cuda.cuCtxCreate(0, device)
    err, = cuda.cuCtxPushCurrent(cuda_context) 
    print_current_ctx(i)

    # What context is this using?
    llm = OfflineLLM(
        model_config=ModelConfig(
            model=args.model,
            tokenizer=None,
            dtype="fp16",
        ),
        disagg_parallel_config=DisaggParallelConfig(
            context=ParallelConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=1
            ),
            decoding=ParallelConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=1
            )
        ),
        cache_config=CacheConfig(
            block_size=16,
            max_num_blocks_per_req=1024,
            gpu_memory_utilization=0.95,
            cpu_swap_space=1.0
        ),
        context_sched_config=ContextStageSchedConfig(
            policy="fcfs",
            max_batch_size=4,
            max_tokens_per_batch=16384
        ),
        decoding_sched_config=DecodingStageSchedConfig(
            policy="fcfs",
            max_batch_size=4,
            max_tokens_per_batch=16384
        )
    )
    print_current_ctx(i)
    print(f"{i}: {torch.cuda.current_device()}")
    del llm

    err, cuda_context = cuda.cuCtxPopCurrent() 
    cuda.cuCtxDestroy(cuda_context)


t0 = threading.Thread(target=thread, args=(0,))
t1 = threading.Thread(target=thread, args=(1,))
t0.start()
t1.start()

t0.join()
t1.join()

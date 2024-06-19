import argparse
from distserve import OfflineLLM, SamplingParams
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
import os
parser = argparse.ArgumentParser()
# MODEL = "/users/xyx/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/"
MODEL = "facebook/opt-125m"
parser.add_argument('--model', type=str, help='The model to use', default=MODEL)
args = parser.parse_args()

mps_dir = '~/xyx/DistServe/examples/mps'
os.environ["CUDA_MPS_PIPE_DIRECTORY"] = f"{mps_dir}/nvidia-mps"
os.environ["CUDA_MPS_LOG_DIRECTORY"] = f"{mps_dir}/nvidia-log"
# Sample prompts.
prompts = [
    "Life blooms like a flower. Far away or by the road. Waiting",
    "A quick brown fox",
    "Artificial intelligence is",
    "To be or not to be,",
    "one two three four"
]

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=4, stop=["\n"]
)

# Create an LLM for offline inference.
llm = OfflineLLM(
    model_config=ModelConfig(
        model=args.model,
        tokenizer=None
    ),
    disagg_parallel_config=DisaggParallelConfig(
        context=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            mps_percentage=0.8,
        ),
        decoding=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            mps_percentage=0.2,
        )
    ),
    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=1024,
        gpu_memory_utilization=0.45,
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

# Generate texts from the prompts. The output is a list of Request objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

# Print the outputs.
for prompt, step_outputs in zip(prompts, outputs):
    # new_token_ids = [step_output.new_token_id for step_output in step_outputs]
    # output_text = llm.tokenizer.decode(new_token_ids)
    print(
        f"Prompt: {prompt!r}, Generated text: {' '.join([step_output.new_token for step_output in step_outputs])} ({len(step_outputs)} tokens generated)."
    )

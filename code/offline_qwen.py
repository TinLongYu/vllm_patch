from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is"
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM object
llm = LLM(model="/home/tim/models/Qwen3-30B-A3B", enforce_eager=True, tensor_parallel_size=4, gpu_memory_utilization=0.9, enable_expert_parallel=True)

# Generate texts from the prompts
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generate_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated Text: {generated_text!r}")
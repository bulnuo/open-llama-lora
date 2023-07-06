import os
import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

#from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

BASE_MODEL = './models/openlm-research/open_llama_3b'
FINETUNED_MODEL = "./models/finetuned-openllama"

device = "cpu"

### 1. Configure Tokenizer for Open-Llama

base_model: str = BASE_MODEL

tokenizer = LlamaTokenizer.from_pretrained(base_model)

### 2. Load pre-trainen Open-LLama model 

model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
            #base_model, device_map="auto", low_cpu_mem_usage=True
        )


### 3. Load fine-tuned model 

lora_weights: str = FINETUNED_MODEL

model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
        #device_map="auto"
        )

### 4.Define and Configure evaluation function 

prompt_template: str = 'nuo-open-llama' 
prompter = Prompter(prompt_template)

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model = torch.compile(model)

def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
    prompt = prompter.generate_prompt(instruction, input)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)

        return prompter.get_response(output)

### 5. Ask questions

def answer(question):
    print("\n\nREQUEST:", question)
    print("RESPONSE:", evaluate(question))

answer("Tell me about alpacas.")
answer("Tell me about Avalora.")
answer("Describe the ancient ruins of Mystoria")
answer("Describe the Gates of Destiny")
answer("Describe the mythical creatures of Avalora's ocean")
answer("Tell me about the Guardians of the Astral Gate")
answer("Describe the Veil of Shadows")


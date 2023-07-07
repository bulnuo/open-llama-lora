# open-llama-lora

## Description

This is a basic example of how to finetune a pre-trained model to be knowledgeable about your domain. 

### What's in the name?
- **LLaMA** is a family of large language models (LLMs) developed by Meta. They are trained on a massive dataset of text and code, and are able to perform a variety of tasks, such as generating text, translating languages, and writing different kinds of creative content.  Llama pre-trained models vary from 7B to 65B parameters
- **OpenLlaMa** is permissively licensed open source reproduction of Meta’s LLaMa developed by OpenLM Research
- **LoRa** is a technique for finetuning LlaMa models. It allows you to train a model on a smaller dataset, while still achieving good performance

### Data
   
For model finetuning, we're using an example dataset that has just 100 records (see `data/custom_avalora.json`). This JSON file is formatted in a way that OpenLlaMa can easily digest.

### Steps

Here's what our process involves:

1. Download the pre-trained OpenLlaMa model, with 3 billion parameters, from Huggingface.
2. Finetune the model using your personalized training data.
3. Run an inference to confirm the finetuned model understands your training data.

### What to expect
We're using a small language model (3 billion parameters), a tiny dataset for finetuning (100 records), doing it in a short amount of time (10 minutes), and running it on a tiny system (a single 8-core CPU). For comparison, training is run on 100s of GPUs for many days.

Despite these limitations, going through the steps will show you that the finetuned model does pick up the information in the training dataset. Its responses might be a bit vague and jumbled, but don't fret! This is just the first step in the exciting journey.


## Installation

### Resources

This setup is tested on *t2.2xlarge* EC2 instance with Amazon Linux – 8vCPU, 32GB memory, 30GB storage.
It does not yet work on Mac. Later I will package it in Docker for easier consumption.

### Steps
1.	Install git and clone the repository
   
`sudo yum install git`

`git clone bulnuo/open-llama-lora`

3.	Follow installation steps outlined in README.txt file
4.	Download the pre-trained base model
   
`cd scripts`

`python download.py --repo_id openlm-research/open_llama_3b --local_dir  ../models`

## Usage

1.	Run the finetuning 

`python finetune-basic.py`

2.	Run the inference

`python infer-basic.py`

## Next steps

Now you can create your own training dataset to run finetunning and inference. Both scripts can be easily modified to pick up your data and store/pick-up finetuned models under different names.

Note that the pre-trained base model does not change. You can have multiple finetuned Lora Adapters for the same base model.

## License and Guidelines

OpenLlama – Apache License -  [https://github.com/openlm-research/open_llama/blob/main/LICENSE](https://github.com/openlm-research/open_llama/blob/main/LICENSE "https://github.com/openlm-research/open_llama/blob/main/LICENSE")

RedPajama training dataset for OpenLlama - [https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T:// "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T")

You should follow appropriate use of AI within 3DS guidelines found here - [https://dsext001-eu1-215dsi0708-3dswym.3dexperience.3ds.com/#community:1648/post:ASqp_lkhS62KHyoADJs5xg](here "https://dsext001-eu1-215dsi0708-3dswym.3dexperience.3ds.com/#community:1648/post:ASqp_lkhS62KHyoADJs5xg")


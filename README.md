# open-llama-lora

## Description

This is a basic example of how to fine-tune a pre-trained model to be knowledgeable about your domain. 

### What's in the name?
- **LLaMA** is a family of large language models (LLMs) developed by Meta. They are trained on a massive dataset of text and code, and are able to perform a variety of tasks, such as generating text, translating languages, and writing different kinds of creative content.  Llama pre-trained models vary from 7B to 65B parameters
- **OpenLlaMa**is permissively licensed open source reproduction of Meta’s LLaMa developed by OpenLM Research
- **LoRa**is a technique for fine-tuning LlaMa models. It allows you to train a model on a smaller dataset, while still achieving good performance

### Data
   
For model fine-tuning, we're using an example dataset that has just 100 records (see `data/custom_avalora.json`). This JSON file is formatted in a way that OpenLlaMa can easily digest.

### Steps

Here's what our process involves:

1. Download the pre-trained OpenLlaMa model, with 3 billion parameters, from Huggingface.
2. Fine-tune the model using your personalized training data.
3. Run an inference to confirm the fine-tuned model understands your training data.

### What to expect
We're using a small language model (3 billion parameters), a tiny dataset for fine-tuning (100 records), doing it in a short amount of time (10 minutes), and running it on a tiny system (a single 8-core CPU). For comparison, training is run on 100s of GPUs for many days.

Despite these limitations, going through the steps will show you that the fine-tuned model does pick up the information in the training dataset. Its responses might be a bit vague and jumbled, but don't fret! This is just the first step in the exciting journey.

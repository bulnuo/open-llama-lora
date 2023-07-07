# open-llama-lora

## Description

This is a basic example of how to fine-tune a pre-trained model to be knowledgeable about your domain. 

### What is in the name?
- **LLaMA** is a family of large language models (LLMs) developed by Meta. They are trained on a massive dataset of text and code, and are able to perform a variety of tasks, such as generating text, translating languages, and writing different kinds of creative content.  Llama pre-trained models vary from 7B to 65B parameters
- **OpenLlaMa**is permissively licensed open source reproduction of Meta’s LLaMa developed by OpenLM Research
- **LoRa**is a technique for fine-tuning LlaMa models. It allows you to train a model on a smaller dataset, while still achieving good performance

### Data

Example data set for model fine-tuning contains only 100 records (see `data/custome_avalora.json`). The JSON file is formatted for easy consumption by OpenLlaMa.   

### Steps

The example goes through the following steps:
1. Download the pre-trained 3B parameters OpenLlaMa model from Huggingface
2. Finetune the model with your own training data set
3. Run inference to validate that the finetuned model is knowledgeable about your training data

### What to expect
The results from this experiment are minimal. We are using a small LLM (3B parameters), fine-tuning it with a tiny data set (100 records), doing it quickly (10 minutes) on a VERY low-end system (one 8-core CPU).

However, if you run through the steps, you will see that the fine-tuned model is clearly aware of the information in the training dataset. It will give mixed up answers that are foggy at best. But fear not, this is the first step in a long and exciting journey.

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text_gpt(prompt, model_name='gpt2', max_length=150):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=max_length, temperature=0.7, top_k=50, top_p=0.95)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Usage
topic = "The impact of artificial intelligence on society"
print(generate_text_gpt(topic))

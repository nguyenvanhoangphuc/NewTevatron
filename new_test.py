import torch
from transformers import LlamaModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

def get_model_with_lora(base_model_name, peft_model_name):
    # Load LoRA configuration
    peft_config = LoraConfig.from_pretrained(peft_model_name)
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    # Load base model with quantization in 8-bit
    base_model = LlamaModel.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map='auto',
        output_hidden_states=True  # Ensure the model returns hidden states
    )
    
    # Apply LoRA to the base model
    model = get_peft_model(base_model, peft_config)
    model.eval()
    return model

# Load tokenizer and model with LoRA
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = get_model_with_lora('meta-llama/Llama-2-7b-hf', 'castorini/repllama-v1-7b-lora-passage')

# Define input query and passage
query = "What is llama?"
title = "Llama"
passage = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."

# Prepare input
inputs = tokenizer(f'query: {query}</s>passage: {title} {passage}</s>', return_tensors='pt')

# Move tensors to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run model to compute embeddings
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    
    # Use last hidden states to get embeddings
    last_hidden_state = outputs.hidden_states[-1][0]
    
    # Get embeddings for query and passage
    query_end = inputs['input_ids'][0].tolist().index(tokenizer.encode('</s>')[0])
    query_embedding = last_hidden_state[query_end - 1]
    passage_embedding = last_hidden_state[-1]
    
    # Normalize embeddings
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
    passage_embedding = torch.nn.functional.normalize(passage_embedding, p=2, dim=0)
    
    # Compute similarity score
    score = torch.dot(query_embedding, passage_embedding)
    print(score.item())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

def get_model_with_lora(base_model_name, peft_model_name):
    # Tải cấu hình LoRA
    peft_config = LoraConfig.from_pretrained(peft_model_name)
    
    # Cấu hình quantization
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Tải mô hình cơ sở với quantization 8-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map='auto',
        output_hidden_states=True  # Đảm bảo mô hình trả về hidden states
    )
    
    # Áp dụng LoRA lên mô hình cơ sở
    model = get_peft_model(base_model, peft_config)
    model.eval()
    return model

# Tải tokenizer và mô hình với LoRA
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = get_model_with_lora('meta-llama/Llama-2-7b-hf', 'castorini/repllama-v1-7b-lora-passage')

# Định nghĩa đầu vào query và passage
query = "What is llama?"
title = "Llama"
passage = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."

# Chuẩn bị đầu vào
inputs = tokenizer(f'query: {query}</s>passage: {title} {passage}</s>', return_tensors='pt')

# Di chuyển tensors lên GPU nếu có sẵn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inputs = {k: v.to(device) for k, v in inputs.items()}

# Chạy mô hình để tính toán embeddings
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    
    # if outputs.hidden_states is None:
    #     print("vao last hidden states")
    #     # Nếu hidden_states vẫn là None, sử dụng last_hidden_state
    #     last_hidden_state = outputs.last_hidden_state[0]
    # else:
    #     print("vao hidden states")
    #     # Sử dụng hidden_states nếu có
    last_hidden_state = outputs.hidden_states[-1][0]
    
    # Lấy embedding cho query và passage
    query_end = inputs['input_ids'][0].tolist().index(tokenizer.encode('</s>')[0])
    query_embedding = last_hidden_state[query_end-1]
    passage_embedding = last_hidden_state[-1]
    
    # Chuẩn hóa embeddings
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
    passage_embedding = torch.nn.functional.normalize(passage_embedding, p=2, dim=0)
    
    # Tính điểm tương đồng
    score = torch.dot(query_embedding, passage_embedding)
    print(score.item())
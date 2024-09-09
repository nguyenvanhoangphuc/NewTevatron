import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers import LlamaModel, PreTrainedModel
from peft import PeftModel, PeftConfig
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(peft_model_name):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config = quantization_config)
    # Configure quantization

    # Load base model with quantization in 8-bit
    base_model = LlamaModel.from_pretrained(
        peft_model_name,
        quantization_config = quantization_config,
        device_map='auto',
        output_hidden_states=True
    )
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, )
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = get_model('meta-llama/Llama-2-7b-hf')

# Define query and passage inputs
# query = "What is llama?"
# title = "Llama"
# passage = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
# query = "How do llamas contribute to the ecosystem in the Andes?"
# title = "Ecological Role of Llamas"
# passage = "Llamas play a significant role in the Andean ecosystem by grazing on native grasses, which helps to maintain the grassland balance. Their droppings act as a natural fertilizer, enriching the soil with essential nutrients. Additionally, llamas are known to have a minimal impact on the terrain compared to other livestock, as their padded feet prevent soil erosion, which is crucial in the fragile Andean environment."
query = "伝統的なイタリアのピザの主な材料は何ですか？"
title = "イタリアンピザの材料"
passage = "ラマは群れで生活する社会的な動物で、重い荷物を長距離運ぶ能力で知られています。主に南アメリカのアンデス地域に生息しており、何千年も前から家畜化されています。ラマはさまざまな鳴き声やボディランゲージ、耳の動きでお互いにコミュニケーションを取ります。"

query_input = tokenizer(f'query: {query}</s>', return_tensors='pt')
passage_input = tokenizer(f'passage: {title} {passage}</s>', return_tensors='pt')

# Run the model forward to compute embeddings and query-passage similarity score
with torch.no_grad():
    # compute query embedding
    query_outputs = model(**query_input)
    query_embedding = query_outputs.last_hidden_state[0][-1]
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

    # compute passage embedding
    passage_outputs = model(**passage_input)
    passage_embeddings = passage_outputs.last_hidden_state[0][-1]
    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=0)

    # compute similarity score
    score = torch.dot(query_embedding, passage_embeddings)
    print(score)

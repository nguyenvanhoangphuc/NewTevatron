from transformers import LlamaModel
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, PeftConfig
from transformers import PreTrainedModel, AutoModel

def load(
        cls,
        model_name_or_path,
        **hf_kwargs,
):
    # Configure quantization
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # Load base model with quantization in 8-bit
    base_model = LlamaModel.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map='auto',
        output_hidden_states=True,  # Ensure the model returns hidden states
        use_cache=False,
        **hf_kwargs  # Pass any additional keyword arguments
    )
    peft_config = LoraConfig(
        base_model_name_or_path=model_name_or_path,
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        inference_mode=True   # False
    )

    hf_model = get_peft_model(base_model, peft_config)

    print("===="*20)
    print("hf_model", hf_model)

    # config = LoraConfig.from_pretrained(model_name_or_path)
    # base_model = LlamaModel.from_pretrained(config.base_model_name_or_path)
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = 0
    # hf_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=config, is_trainable=True)
    # hf_model = hf_model.merge_and_unload()
    model = cls(
        lm_q=hf_model,
        lm_p=hf_model,
        pooler=None,
        untie_encoder=False
    )
    return model


model_name_or_path = "model_repllama/checkpoint-10"  # Đường dẫn tới thư mục checkpoint

hf_model = load(cls=AutoModel, model_name_or_path=model_name_or_path)

# Thực hiện inference
tokenizer_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name if tokenizer_name else model_name_or_path,
    cache_dir=None
)
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"
inputs = tokenizer("Your query here", return_tensors="pt")
outputs = hf_model(**inputs)

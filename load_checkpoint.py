from transformers import LlamaModel
from peft import PeftModel, PeftConfig

def load_checkpoint(model_name_or_path, checkpoint_path, **hf_kwargs):
    # Load Peft configuration
    peft_config = PeftConfig.from_pretrained(checkpoint_path)

    # Load base model
    base_model = LlamaModel.from_pretrained(
        model_name_or_path,
        device_map='auto',
        output_hidden_states=True,
        use_cache=False,
        **hf_kwargs
    )

    # Apply quantization if necessary
    base_model.quantization_config = peft_config.quantization_config

    # Load the adapter model
    hf_model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False  # Set to False for inference
    )

    # Ensure the pad token is set
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = 0

    # Set the model to evaluation mode
    hf_model.eval()

    return hf_model


checkpoint_path = "path/to/your/checkpoint/directory"  # Đường dẫn tới thư mục checkpoint
model_name_or_path = "model_name_or_path"  # Tên hoặc đường dẫn tới mô hình gốc của bạn

hf_model = load_checkpoint(model_name_or_path, checkpoint_path)

# Thực hiện inference
tokenizer =
inputs = tokenizer("Your query here", return_tensors="pt")
outputs = hf_model(**inputs)

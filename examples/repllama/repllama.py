import torch
import torch.nn as nn
from torch import Tensor
from transformers import LlamaModel, PreTrainedModel
import logging
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from tevatron.modeling.encoder import EncoderModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class RepLLaMA(EncoderModel):
    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False
                 ):
        super().__init__(lm_q, lm_p, pooler, untie_encoder, negatives_x_device)
        self.config = lm_q.config

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, output_hidden_states=True)
        p_hidden = psg_out.hidden_states[-1]
        attention_mask = psg['attention_mask']
        # p_reps is the last token representation that is not padding
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        p_reps = p_hidden[torch.arange(p_hidden.size(0)), last_token_indices]
        p_reps = nn.functional.normalize(p_reps, p=2, dim=-1)
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, output_hidden_states=True)
        q_hidden = qry_out.hidden_states[-1]
        attention_mask = qry['attention_mask']
        # q_reps is the last token representation that is not padding
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        q_reps = q_hidden[torch.arange(q_hidden.size(0)), last_token_indices]
        q_reps = nn.functional.normalize(q_reps, p=2, dim=-1)
        return q_reps


    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1)) / 0.01
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.lm_q.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.lm_p.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    
    @staticmethod
    def build_peft_model(peft_model_name):
        config = LoraConfig.from_pretrained(peft_model_name)
        config.inference_mode = False
        base_model = LlamaModel.from_pretrained(config.base_model_name_or_path)
        model = get_peft_model(base_model, config)
        model.print_trainable_parameters()
        return model
    @classmethod
    def build(
            cls,
            model_args,
            train_args,
            **hf_kwargs,
    ):
        print("====" * 20)
        print("1.x")
        # Configure quantization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        
        # Load base model with quantization in 8-bit
        base_model = LlamaModel.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map='auto',
            output_hidden_states=True,  # Ensure the model returns hidden states
            use_cache=False,
            **hf_kwargs  # Pass any additional keyword arguments
        )
        print("****" * 20)

        if train_args.gradient_checkpointing:
            base_model.enable_input_require_grads()
        
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        peft_config = LoraConfig(
            base_model_name_or_path=model_args.model_name_or_path,
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            inference_mode=False
        )

        hf_model = get_peft_model(base_model, peft_config)

        model = cls(
            lm_q=hf_model,
            lm_p=hf_model,
            pooler=None,
            untie_encoder=False
        )
        return model

    # @classmethod
    # def build(
    #         cls,
    #         model_args,
    #         train_args,
    #         **hf_kwargs,
    # ):
    #     print("===="*20)
    #     print("1.x")
    #     # Configure quantization
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
    #     # Load base model with quantization in 8-bit
    #     base_model = LlamaModel.from_pretrained(
    #         model_args.model_name_or_path,
    #         quantization_config=quantization_config,
    #         device_map='auto',
    #         output_hidden_states=True,  # Ensure the model returns hidden states
    #         use_cache=False
    #     )
    #     print("****"*20)
    #     if train_args.gradient_checkpointing:
    #         base_model.enable_input_require_grads()
        
    #     if base_model.config.pad_token_id is None:
    #         base_model.config.pad_token_id = 0

    #     peft_config = LoraConfig(
    #         base_model_name_or_path=model_args.model_name_or_path,
    #         task_type=TaskType.FEATURE_EXTRACTION,
    #         r=8,
    #         lora_alpha=16,
    #         lora_dropout=0.1,
    #         target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    #         inference_mode=False
    #     )

    #     hf_model = get_peft_model(base_model, peft_config)

    #     model = cls(
    #         lm_q=hf_model,
    #         lm_p=hf_model,
    #         pooler=None,
    #         untie_encoder=False
    #     )
    #     return model
    # @classmethod
    # def build(
    #         cls,
    #         model_args,
    #         train_args,
    #         **hf_kwargs,
    #     ):
    #     # Khởi tạo mô hình cơ bản
    #     base_model = LlamaModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        
    #     # Sử dụng gradient checkpointing để tiết kiệm bộ nhớ trong huấn luyện
    #     if train_args.gradient_checkpointing:
    #         base_model.gradient_checkpointing_enable()  # Sử dụng phương thức tối ưu hơn
    #         base_model.enable_input_require_grads()

    #     # Thiết lập giá trị pad_token_id nếu chưa được thiết lập
    #     if base_model.config.pad_token_id is None:
    #         base_model.config.pad_token_id = 0

    #     # Cấu hình LoRA để tối ưu bộ nhớ
    #     peft_config = LoraConfig(
    #         base_model_name_or_path=model_args.model_name_or_path,
    #         task_type=TaskType.FEATURE_EXTRACTION,
    #         r=2,  # Giảm rank để tiết kiệm bộ nhớ
    #         lora_alpha=8,  # Giảm alpha
    #         lora_dropout=0.1,  # Giữ nguyên dropout để đảm bảo không overfitting
    #         target_modules=["q_proj", "v_proj"],  # Chọn ít module hơn để fine-tune
    #         inference_mode=True  # Bật chế độ inference để tiết kiệm bộ nhớ
    #     )

    #     # Áp dụng cấu hình LoRA vào mô hình cơ bản
    #     hf_model = get_peft_model(base_model, peft_config)

    #     # Khởi tạo lớp với mô hình được tối ưu
    #     model = cls(
    #         lm_q=hf_model,
    #         lm_p=hf_model,
    #         pooler=None,
    #         untie_encoder=False
    #     )

    #     return model

    
    @classmethod
    def load(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = LlamaModel.from_pretrained(config.base_model_name_or_path)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        hf_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=config, is_trainable=True)
        hf_model = hf_model.merge_and_unload()
        model = cls(
            lm_q=hf_model,
            lm_p=hf_model,
            pooler=None,
            untie_encoder=False
        )
        return model

    def save(self, output_dir: str):
        self.lm_q.save_pretrained(output_dir)

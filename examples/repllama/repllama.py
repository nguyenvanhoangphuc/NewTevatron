import torch
import torch.nn as nn
from torch import Tensor
from transformers import LlamaModel, PreTrainedModel
import logging
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, PeftConfig
from tevatron.modeling.encoder import EncoderModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import math
import os

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
        # print("===="*20)
        # print(self.lm_p.device)
        # self.lm_p.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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
        print("===="*20)
        print("qry", qry)
        qry_out = self.lm_q(**qry, output_hidden_states=True)
        # print("qry_out", qry_out)
        # print("qry_out.last_hidden_state.shape", qry_out.last_hidden_state.shape)
        # print("len(qry_out.hidden_states)", len(qry_out.hidden_states))
        # print("qry_out.hidden_states.shape", qry_out.hidden_states[-1].shape)
        # print("qry_out.last_hidden_state", qry_out.last_hidden_state)
        # print("qry_out.hidden_states", qry_out.hidden_states[-1])
        q_hidden = qry_out.hidden_states[-1]
        # print("q_hidden", q_hidden)
        # if math.isnan(q_hidden[0]): 
        #     # print("q_hidden[0]", q_hidden[0])
        #     print("q_hidden[0].shape", q_hidden[0].shape)
        
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
        print(cls)
        # Configure quantization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        
        # Load base model with quantization in 8-bit
        base_model = LlamaModel.from_pretrained(
            os.path.join(os.getcwd(), "model_repllama/checkpoint-10"),
            quantization_config = quantization_config
        )
        print("****" * 20)

        if train_args.gradient_checkpointing:
            base_model.enable_input_require_grads()
        
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        # peft_config = LoraConfig(
        #     base_model_name_or_path=os.path.join(os.getcwd(), "model_repllama/checkpoint-14600"), #model_args.model_name_or_path
        #     task_type=TaskType.FEATURE_EXTRACTION,
        #     r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        #     inference_mode=False
        # )
        
        config = LoraConfig.from_pretrained(os.path.join(os.getcwd(), "model_repllama/checkpoint-10"))


        hf_model = PeftModel.from_pretrained(base_model, os.path.join(os.getcwd(), "model_repllama/checkpoint-10"), config=config, is_trainable=True)
        print("this")

        # hf_model = hf_model.merge_and_unload()


        print("===="*20)
        print("hf_model", hf_model)

        # In tất cả các tham số của mô hình
        for name, param in hf_model.named_parameters():
            print(f"Parameter Name: {name}")
            print(f" - Shape: {param.shape}")
            print(f" - Requires Grad: {param.requires_grad}")
            print(f" - Values: {param.data}\n")

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
        # Configure quantization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        
        # Load base model with quantization in 8-bit
        base_model = LlamaModel.from_pretrained(
            os.path.join(os.getcwd(), "model_repllama/checkpoint-10"),
            quantization_config = quantization_config
        )
        print("****" * 20)
        
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        # peft_config = LoraConfig(
        #     base_model_name_or_path=os.path.join(os.getcwd(), "model_repllama/checkpoint-14600"), #model_args.model_name_or_path
        #     task_type=TaskType.FEATURE_EXTRACTION,
        #     r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        #     inference_mode=False
        # )
        
        config = LoraConfig.from_pretrained(os.path.join(os.getcwd(), "model_repllama/checkpoint-10"))


        hf_model = PeftModel.from_pretrained(base_model, os.path.join(os.getcwd(), "model_repllama/checkpoint-10"), config=config, is_trainable=True)
        print("this")

        # hf_model = hf_model.merge_and_unload()

        print("===="*20)
        print("hf_model", hf_model)

        # In tất cả các tham số của mô hình
        for name, param in hf_model.named_parameters():
            print(f"Parameter Name: {name}")
            print(f" - Shape: {param.shape}")
            print(f" - Requires Grad: {param.requires_grad}")
            print(f" - Values: {param.data}\n")
        model = cls(
            lm_q=hf_model,
            lm_p=hf_model,
            pooler=None,
            untie_encoder=False
        )
        return model

    def save(self, output_dir: str):
        print("self.lm_q.save_pretrained(output_dir)")
        # In tất cả các tham số của mô hình
        for name, param in self.lm_q.named_parameters():
            print(f"Parameter Name: {name}")
            print(f" - Shape: {param.shape}")
            print(f" - Requires Grad: {param.requires_grad}")
            print(f" - Values: {param.data}\n")
        self.lm_q.save_pretrained(output_dir)

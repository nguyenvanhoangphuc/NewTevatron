#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Chạy train 
# openai-community/gpt2
# mistralai/Mistral-7B-v0.1
# python src/tevatron/retriever/driver/train.py --output_dir retriever-mistral --model_name_or_path mistralai/Mistral-7B-v0.1 --lora --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj --save_steps 50 --dataset_name Tevatron/msmarco-passage-aug --query_prefix "Query: " --passage_prefix "Passage: " --fp16 --pooling eos --append_eos_token --normalize --temperature 0.01 --per_device_train_batch_size 2 --gradient_checkpointing --train_group_size 2 --learning_rate 1e-4 --query_max_len 32 --passage_max_len 156 --num_train_epochs 1 --logging_steps 10 --overwrite_output_dir --gradient_accumulation_steps 2
python examples/repllama/train.py --output_dir model_repllama --model_name_or_path meta-llama/Llama-2-7b-hf --save_steps 200 --dataset_name Tevatron/msmarco-passage --bf16 --per_device_train_batch_size 4 --gradient_accumulation_steps 2 --gradient_checkpointing --train_n_passages 16 --learning_rate 1e-4 --q_max_len 32 --p_max_len 196 --num_train_epochs 1 --logging_steps 10 --overwrite_output_dir --dataset_proc_num 32 --negatives_x_device --warmup_steps 100 >> training.log
# EMBEDDING_OUTPUT_DIR=EmbeddingOutput

# python -m tevatron.retriever.driver.encode --output_dir=temp --model_name_or_path retriever-mistral/checkpoint-3836 --lora_name_or_path retriever-mistral --lora --query_prefix "Query: " --passage_prefix "Passage: " --bf16 --pooling eos --append_eos_token --normalize --encode_is_query --per_device_eval_batch_size 128 --query_max_len 32 --passage_max_len 156 --dataset_name Tevatron/msmarco-passage --dataset_split dev --encode_output_path $EMBEDDING_OUTPUT_DIR/query-dev.pkl
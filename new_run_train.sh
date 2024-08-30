#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Chạy train 
mkdir beir_embedding_scifact
# castorini/repllama-v1-7b-lora-passage
# meta-llama/Llama-2-7b-hf
# /home/thanhnguyen/Data/Model_results/LinhModel/Rakuten-7b-cross-eval_5eps_evalloss/checkpoint/Rakuten-7b-cross-eval_5eps_evalloss
# Rakuten/RakutenAI-7B
python examples/repllama/encode.py --output_dir temp --model_name_or_path castorini/repllama-v1-7b-lora-passage --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 1 --p_max_len 512 --dataset_name Tevatron/beir-corpus:scifact --encoded_save_path beir_embedding_scifact/corpus_scifact.pkl --encode_num_shard 1 --encode_shard_index 0
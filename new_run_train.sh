#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=1

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Chạy train 
# castorini/repllama-v1-7b-lora-passage
# meta-llama/Llama-2-7b-hf
# /home/thanhnguyen/Data/Model_results/LinhModel/Rakuten-7b-cross-eval_5eps_evalloss/checkpoint/Rakuten-7b-cross-eval_5eps_evalloss
# Rakuten/RakutenAI-7B

# # Chạy encode corpus
mkdir beir_embedding_scifact_new
# # castorini/repllama-v1-7b-lora-passage
# # tevatron/model_repllama/checkpoint-14200
# python examples/repllama/encode.py --output_dir temp --model_name_or_path model_repllama/checkpoint-14200 --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 1 --p_max_len 512 --dataset_name Tevatron/beir-corpus:scifact --encoded_save_path beir_embedding_scifact/corpus_scifact.pkl --encode_num_shard 1 --encode_shard_index 0 >> encode.log

# Chạy encode query
# castorini/repllama-v1-7b-lora-passage
# => tevatron/model_repllama/checkpoint-14200
# meta-llama/Llama-2-7b-hf  
# => tevatron/model_repllama/checkpoint-14200
# Tevatron/beir:scifact/test
# => test_retrieval_ms_marco_dataset
python examples/repllama/encode.py --output_dir temp --model_name_or_path model_repllama/checkpoint-10 --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 4 --q_max_len 64 --p_max_len 196 --dataset_name test_retrieval_ms_marco_dataset --encoded_save_path beir_embedding_scifact_new/queries_scifact.pkl --encode_is_qry > encode_q.log
# python examples/repllama/encode.py --output_dir temp --model_name_or_path model_repllama/checkpoint-200 --tokenizer_name meta-llama/Llama-2-7b-hf --fp16 --per_device_eval_batch_size 1 --p_max_len 512 --dataset_name Tevatron/beir:scifact/test --encoded_save_path beir_embedding_scifact/queries_scifact.pkl --encode_is_qry

# # Search
# python -m tevatron.faiss_retriever --query_reps beir_embedding_scifact/queries_scifact.pkl --passage_reps 'beir_embedding_scifact/corpus_scifact.pkl' --depth 100 --batch_size 64 --save_text --save_ranking_to beir_embedding_scifact/rank.scifact.txt
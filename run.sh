#!/bin/bash

#建立数据库
python build_db.py --dataset_path convert_hitab/hitab/hitab_test.jsonl --max_encode_cell 10000

#尝试处理前5个
python run.py --stop_at 5 --verbose --dataset_path convert_hitab/hitab/hitab_test.jsonl  --model_name qwen_3b --embed_model_name sentence-transformers/all-MiniLM-L6-v2

 python run.py --stop_at 5 --verbose --dataset_path convert_hitab/hitab/hitab_test.jsonl  --model_name tablellm_7b --embed_model_name sentence-transformers/all-MiniLM-L6-v2


 python run.py --dataset_path convert_hitab/hitab/hitab_test.jsonl --model_name qwen_3b --embed_model_name sentence-transformers/all-MiniLM-L6-v2  --agent_type TableRAG --log_dir 'output/hitab_qwen3_tablerag' --top_k 5 --sc 10 --max_encode_cell 100 --n_worker 16

 python run.py  --log_dir 'output/hitab_qwen3_tablerag' --dataset_path convert_hitab/hitab/hitab_test.jsonl  --model_name qwen_3b --embed_model_name sentence-transformers/all-MiniLM-L6-v2  --top_k 5 --sc 10 --max_encode_cell 100 --n_worker 4 --stop_at 50
python collect_feat.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 4096 --max_batch_size 6

python collect_feat_sacred.py -F train with model.max_seq_len=4096 model.max_batch_size=6
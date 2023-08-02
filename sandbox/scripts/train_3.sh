# CUDA_VISIBLE_DEVICES=3 python -m sandbox.main --model_name seq2seq_attn  --out_dim 2 --epochs 100 --do_jitter --exp_name seq2seq_jitter_attn_trial1
# CUDA_VISIBLE_DEVICES=3 python -m sandbox.main --model_name seq2seq_attn  --out_dim 2 --epochs 100 --do_jitter --exp_name seq2seq_jitter_attn_trial2
# CUDA_VISIBLE_DEVICES=3 python -m sandbox.main --model_name seq2seq_attn  --out_dim 2 --epochs 100 --do_jitter --exp_name seq2seq_jitter_attn_trial3
###
CUDA_VISIBLE_DEVICES=3 python -m sandbox.main --model_name seq2seq_attn  --out_dim 2 --epochs 100 --do_mixup --exp_name seq2seq_mixup_attn_trial1
CUDA_VISIBLE_DEVICES=3 python -m sandbox.main --model_name seq2seq_attn  --out_dim 2 --epochs 100 --do_mixup --exp_name seq2seq_mixup_attn_trial2
CUDA_VISIBLE_DEVICES=3 python -m sandbox.main --model_name seq2seq_attn  --out_dim 2 --epochs 100 --do_mixup --exp_name seq2seq_mixup_attn_trial3
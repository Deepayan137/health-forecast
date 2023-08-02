# CUDA_VISIBLE_DEVICES=2 python -m sandbox.main --model_name seq2seq  --out_dim 2 --epochs 100 --do_jitter --exp_name seq2seq_jitter_trial1
# CUDA_VISIBLE_DEVICES=2 python -m sandbox.main --model_name seq2seq  --out_dim 2 --epochs 100 --do_jitter --exp_name seq2seq_jitter_trial2
# CUDA_VISIBLE_DEVICES=2 python -m sandbox.main --model_name seq2seq  --out_dim 2 --epochs 100 --do_jitter --exp_name seq2seq_jitter_trial3
###
CUDA_VISIBLE_DEVICES=2 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 100 --do_mixup --exp_name lstm_mixup_trial1
CUDA_VISIBLE_DEVICES=2 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 100 --do_mixup --exp_name lstm_mixup_trial2
CUDA_VISIBLE_DEVICES=2 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 100 --do_mixup --exp_name lstm_mixup_trial3
###
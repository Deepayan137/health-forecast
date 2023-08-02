CUDA_VISIBLE_DEVICES=1 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 100 --exp_name lstm_trial1
CUDA_VISIBLE_DEVICES=1 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 100 --exp_name lstm_trial2
CUDA_VISIBLE_DEVICES=1 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 100 --exp_name lstm_trial3
###
CUDA_VISIBLE_DEVICES=1 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 150 --lr 0.0001 --exp_name lstm_lr_0.0001_trial1
CUDA_VISIBLE_DEVICES=1 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 150 --lr 0.0001 --exp_name lstm_lr_0.0001_trial1
CUDA_VISIBLE_DEVICES=1 python -m sandbox.main --model_name lstm  --out_dim 2 --epochs 150 --lr 0.0001 --exp_name lstm_lr_0.0001_trial1

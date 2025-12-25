export CUDA_VISIBLE_DEVICES=2,3
torchrun --standalone --nnodes 1 --nproc-per-node 2 main.py fit \
    --config config/lam-stage-2.yaml \
    > logs/lam-stage-2.log 2>&1 &
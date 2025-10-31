export OMP_NUM_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=.:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

torchrun --nnodes=1 --nproc_per_node=1 libero_dataset.py \
--seed 42 \
--vlm_path /home/linyihan/linyh/VLM/InternVL3_5-1B \
--data_root_dir /home/linyihan/linyh/datasets/modified_libero_rlds/libero_object_no_noops/1.0.0 \
--data_mix libero_object \
--window_size 8 \
--max_steps 10000 \
--per_device_batch_size 32 \
#> logs/libero-object--internvl-pro--$current_time.log 2>&1 &

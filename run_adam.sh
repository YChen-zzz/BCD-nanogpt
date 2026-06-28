# source /root/miniconda3/etc/profile.d/conda.sh

# conda env list
# conda activate llm_test

# source /usr/local/Ascend/ascend-toolkit/set_env.sh

# echo "===== python ====="
# which python3
# python3 -V
# python3 -c "import sys; print(sys.executable)"
# python3 -c "import torch; print(\"torch\", torch.__version__, torch.__file__)"
# python3 -c "import torch_npu; print(\"torch_npu\", torch_npu.__version__, torch_npu.__file__)"

torchrun --standalone --nproc_per_node=8 train_gpt2.py \
    --model 124m \
    --optimizer adamw \
    --seed 42 \
    --chinchilla_multiplier 1.0 \
    --lr 0.001 \
    --beta1 0.9 \
    --beta2 0.95 \
    --sequence_length 2048 \
    --eps 1.0e-8 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --batch_size 512 \
    --warmup_fraction 0.05 \
    --wsd_fraction 0.2 \
    --output_dir test_model_files/try1_fused_adamw_130m_270_100B \
    "$@"

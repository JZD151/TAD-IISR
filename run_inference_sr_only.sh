set -e

OUT_DIR="./output/TAD-IISR-SR/ITSR-testset"

mkdir -p "$OUT_DIR"

accelerate launch --num_processes=1 --gpu_ids="0" --main_process_port 29300 src/inference_tadiisr_sr_only.py \
    --sd_path="The path of the downloaded SD-Turbo" \
    --pretrained_path="./output/checkpoints/model_50001.pkl" \
    --de_enc_path="./output/checkpoints/de_enc_50001.pkl" \
    --output_dir="$OUT_DIR" \
    --mixed_precision="fp16" \
    --align_method="nofix" \
    --val_batch_size=1

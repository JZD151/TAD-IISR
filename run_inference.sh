set -e

OUT_DIR="./output/TAD-IISR/ITSR-testset"
REDE_DIR="./output/TAD-IISR/ITSR-testset_redegraded"

mkdir -p "$OUT_DIR" "$REDE_DIR"

accelerate launch --num_processes=1 --gpu_ids="0" --main_process_port 29300 src/inference_tadiisr.py \
    --sd_path="The path of the downloaded SD-Turbo" \
    --pretrained_path="./output/checkpoints/model_50001.pkl" \
    --de_enc_path="./output/checkpoints/de_enc_50001.pkl" \
    --re_de_path="./output/checkpoints/re_de_50001.pkl" \
    --save_redegraded \
    --redegraded_dir="$REDE_DIR" \
    --output_dir="$OUT_DIR" \
    --mixed_precision="fp16" \
    --align_method="nofix"

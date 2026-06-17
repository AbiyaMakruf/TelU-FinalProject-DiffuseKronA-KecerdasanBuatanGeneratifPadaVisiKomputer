export adapter_type="krona"
export attn_update_unet="kqvo"
export prompt="a watercolour painting of a photo of skshuman_nityanand with mountains in the background high quality, 4k"
export checkpoint_path="../outputs/human_nityanand/krona_k64:8q64:8v64:8o64:8_sdxl_0.001/checkpoint-500"

# dog6 subject images are available at dataset link provided in the README
accelerate launch inference_sdxl.py \
    --checkpoint_path=$checkpoint_path \
    --output_path=$checkpoint_path \
    --adapter_type=$adapter_type \
    --attn_update_unet=$attn_update_unet \
    --prompt="$prompt" \
    --seed=0

export adapter_type="krona"
export attn_update_unet="kqvo"
export prompt="A skshuman_abiya person in an astronaut suit in a spaceship"
# export prompt="A skshuman_abiya person taking a shot in basketball"
# export prompt="A skshuman_abiya person standing under the pink blossoms of a cherry tree"
export checkpoint_path="../outputs/human_abiya/krona_k64:8q64:8v64:8o64:8_sdxl_0.001/checkpoint-1000"

# human_abiya subject images are available at dataset link provided in the README
accelerate launch inference_sdxl.py \
    --checkpoint_path=$checkpoint_path \
    --output_path=$checkpoint_path \
    --adapter_type=$adapter_type \
    --attn_update_unet=$attn_update_unet \
    --prompt "$prompt" \
    --seed=0

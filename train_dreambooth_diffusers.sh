export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATASET_NAME="/workspace/dataset/train2"
export OUTPUT_DIR="my-Flux-LoRA-v5"

#accelerate launch train_dreambooth_lora_flux_advanced.py \
accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --instance_prompt="photo of ohwx man standing in a studio setting" \
  --output_dir=$OUTPUT_DIR \
  --caption_column="text" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --repeats=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --optimizer="prodigy"\
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=700 \
  --checkpointing_steps=10 \
  --seed="0" \
  --resume_from_checkpoint=latest \
  --validation_prompt "photo of ohwx man standing in a studio setting" \
  --validation_epochs=1000



  #--resume_from_checkpoint=latest \
  #--token_abstraction="ohwx", \
  #--train_text_encoder_ti\
  #--train_text_encoder_ti_frac=0.5\
  #--text_encoder_lr=1.0 \
  #--enable_t5_ti\
  #--initializer_concept="boy"

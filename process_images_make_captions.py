from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import torch
import os
import glob
import csv
import math
from PIL import Image
from qwen_vl_utils import process_vision_info

def resize_with_aspect(image, max_size=1280):
    """Resize image preserving aspect ratio so largest dimension is max_size"""
    width, height = image.size
    
    if width >= max_size or height >= max_size:
        # Calculate scaling factor
        scale = max_size / max(width, height)
        new_width = math.ceil(width * scale)
        new_height = math.ceil(height * scale)
    
    # Round dimensions to nearest multiple of 28
    new_width = round(width / 28) * 28
    new_height = round(height / 28) * 28
    
    # Ensure dimensions are at least 28
    new_width = max(28, new_width)
    new_height = max(28, new_height)
    
    return image.resize((new_width, new_height), Image.LANCZOS)

# Create processed directory if it doesn't exist
processed_dir = "processed"
os.makedirs(processed_dir, exist_ok=True)

# Load the model
print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2.5-VL-32B-Instruct",
     torch_dtype=torch.bfloat16,
     #attn_implementation="flash_attention_2",
     device_map="auto",
)

# Load the processor
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)
# Find all image files in the dataset directory
image_patterns = [
    "./input/*.*"
]
image_files = []
for pattern in image_patterns:
    image_files.extend(glob.glob(pattern))

print(f"Found {len(image_files)} images to process.")

# Process each image
for img_path in image_files:
    print(f"Processing {img_path}...")
    try:
        # Load the image
        image = Image.open(img_path)
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": resize_with_aspect(image)
                    },
                    {
                        "type": "text", 
                        "text": "Describe the content of the image as one sentence. Replace any mention of man and his hair with the word \"MARK\". Focus on other visual elements such as objects, background, colors, and overall scene. Start with \"a photo of\"."
                    }
                ], 
            }
        ]

        # Prepare for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate caption
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]  # Get first element since we're processing one image at a time
        
        # Get filename and create individual text file
        filename = Path(img_path).name
        txt_filename = Path(filename).stem + ".txt"
        txt_path = os.path.join(processed_dir, txt_filename)
        
        # Save caption to individual text file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        
        # Save image as JPG with 95% compression
        new_filename = Path(filename).stem + ".jpg"
        output_path = os.path.join(processed_dir, new_filename)
        image = image.convert("RGB")  # Convert to RGB (in case of PNG with alpha channel)
        image.save(output_path, "JPEG", quality=95)
        
        print(f"  Caption: {output_text[:50]}...")
        print(f"  Saved as: {output_path}")
        print(f"  Text saved as: {txt_path}")
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")

print(f"Processing complete. Results saved to {processed_dir} folder.") 
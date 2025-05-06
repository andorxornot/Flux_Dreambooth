import os
import glob
import json
from PIL import Image

def process_files():
    # Create dataset/train directory if it doesn't exist
    os.makedirs('dataset/train', exist_ok=True)
    
    # Get all .txt files in the processed/ directory
    txt_files = glob.glob('processed/*.txt')
    
    if not txt_files:
        print("No .txt files found in the 'processed/' directory.")
        return
    
    # Get all non-txt, non-jsonl image files for matching
    image_files = [f for f in glob.glob('processed/*.*') 
                  if not f.endswith(('.txt'))]
    image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    
    metadata_entries = []
    
    # Process each text file
    for idx, txt_file in enumerate(txt_files):
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        
        # Find matching image or default to jpg
        image_path = image_dict.get(base_name, f"processed/{base_name}.jpg")
        
        try:
            # Resize and save image
            with Image.open(image_path) as img:
                # Resize if any dimension > 1280
                width, height = img.size
                if width > 1280 or height > 1280:
                    # Calculate the scaling factor
                    scale = 1280 / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save resized image to dataset/train directory
                image_name = f"{idx}.jpg"
                new_image_path = f"dataset/train/{image_name}"
                img.save(new_image_path)
            
            # Read and process the text file
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().replace("MARK", "p3rs0n").strip()

            metadata_entries.append({
                "file_name": image_name,
                "text": content
            })
            
        except Exception as e:
            print(f"Error processing {txt_file} or {image_path}: {e}")
    
    # Write metadata to jsonl file
    with open('dataset/train/metadata.jsonl', 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Successfully created metadata.jsonl with {len(metadata_entries)} entries.")
    print(f"Resized and saved {len(metadata_entries)} images to dataset/train/")

if __name__ == "__main__":
    process_files() 
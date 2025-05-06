import os
import glob
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Directory path
processed_dir = "/workspace/processed3"

# Load YOLO model
model = YOLO("yolov8n.pt")  # using the smallest model for efficiency

# Get all JPG files in the directory
jpg_files = glob.glob(os.path.join(processed_dir, "*.jpg"))
print(f"Found {len(jpg_files)} JPG files in {processed_dir}")

# Process each file
for file_path in jpg_files:
    try:
        # Open the image
        #file_path = "/workspace/processed3/IMG_6994 2.jpg"
        img = Image.open(file_path)
        width, height = img.size
        print(f"\n--- Processing {os.path.basename(file_path)} ---")
        print(f"Original dimensions: {width}x{height}")
        
        # Check dimensions
        if width < 1024 or height < 1024:
            print(f"Deleting {file_path} (dimensions: {width}x{height})")
            os.remove(file_path)
            continue
        
        # Convert to numpy array for YOLO
        img_array = np.array(img)
        
        # Run YOLO detection
        results = model(img_array, classes=0)  # class 0 is person in YOLO
        
        # Check if any person was detected
        if len(results[0].boxes) == 0:
            print(f"No person detected in {file_path}")
            os.remove(file_path)
            continue
        
        # Get the person with the largest bounding box (assumed to be the main person)
        boxes = results[0].boxes
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        main_person_idx = areas.argmax().item()
        
        # Get the bounding box coordinates
        box = boxes.xyxy[main_person_idx].cpu().numpy()
        x1, y1, x2, y2 = box
        print(f"Person box [x1, y1, x2, y2]: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # Calculate the center of the person
        person_center_x = (x1 + x2) / 2
        person_center_y = (y1 + y2) / 2
        print(f"Person center: ({person_center_x:.1f}, {person_center_y:.1f})")

        # Get person width and height
        person_width = (x2 - x1) * 1.2
        person_height = (y2 - y1) * 1.2
        print(f"Person dimensions: {person_width:.1f}x{person_height:.1f}")

        # Determine the desired side length for the square crop
        # Use the larger dimension of the person's bounding box
        desired_side_length = max(person_width, person_height)
        print(f"Desired crop side length (max(person_w, person_h)): {desired_side_length:.1f}")

        # Ensure the side length is not larger than the image dimensions
        final_side_length = min(desired_side_length, width, height)
        print(f"Final crop side length (min(desired, img_w, img_h)): {final_side_length:.1f}")

        # Calculate the ideal top-left corner coordinates for the square crop, centered on the person
        ideal_x1 = person_center_x - final_side_length / 2
        ideal_y1 = person_center_y - final_side_length / 2
        print(f"Ideal top-left corner (center - side/2): ({ideal_x1:.1f}, {ideal_y1:.1f})")

        # Adjust coordinates to ensure the square crop stays within image boundaries
        # Clamp x1 to be at least 0
        crop_x1 = max(0, ideal_x1)
        # Clamp y1 to be at least 0
        crop_y1 = max(0, ideal_y1)
        print(f"Clamped top-left (max(0, ideal)): ({crop_x1:.1f}, {crop_y1:.1f})")

        # Further adjust x1 and y1 to ensure the bottom-right corner is within bounds
        # If x1 + side > width, move x1 left to width - side. Clamp at 0.
        crop_x1 = max(0, min(crop_x1, width - final_side_length))
        # If y1 + side > height, move y1 up to height - side. Clamp at 0.
        crop_y1 = max(0, min(crop_y1, height - final_side_length))
        print(f"Final adjusted top-left (to fit bottom-right): ({crop_x1:.1f}, {crop_y1:.1f})")
        
        # Recalculate bottom-right corner based on the final adjusted top-left and side length
        crop_x2 = crop_x1 + final_side_length
        crop_y2 = crop_y1 + final_side_length
        print(f"Final bottom-right corner (float): ({crop_x2:.1f}, {crop_y2:.1f})")

        # Calculate final integer coordinates ensuring squareness
        int_x1 = int(round(crop_x1)) # Round might be better than truncating with int()
        int_y1 = int(round(crop_y1))
        # Ensure side length calculation maintains squareness after int conversion
        # We use the float side length for potentially better centering/fitting before int
        # but calculate the int box size consistently
        # Let's recalculate int_side based on potentially rounded int_x1/y1
        # No, let's stick to the calculated final_side_length and ensure the box reflects that.
        int_side = int(round(final_side_length))

        # Adjust int_x1, int_y1 again slightly if rounding pushes box out of bounds
        int_x1 = max(0, min(int_x1, width - int_side))
        int_y1 = max(0, min(int_y1, height - int_side))

        int_x2 = int_x1 + int_side
        int_y2 = int_y1 + int_side
        crop_box = (int_x1, int_y1, int_x2, int_y2)
        print(f"Final integer crop_box (calculated for squareness): {crop_box}")

        # Crop the image
        cropped_img = img.crop(crop_box)
        print(f"Cropped image dimensions: {cropped_img.size[0]}x{cropped_img.size[1]}")

        # Check if cropped image is actually square before resizing
        if cropped_img.size[0] != cropped_img.size[1]:
            print(f"ERROR: Cropped image is NOT square for {os.path.basename(file_path)} - Size: {cropped_img.size[0]}x{cropped_img.size[1]}")
            # Optionally skip saving or raise an error
            # continue 

        # Resize the cropped image to 1024x1024
        # Since the crop_box is guaranteed to be square, resizing maintains aspect ratio
        resized_img = cropped_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        print(f"Resized image dimensions: {resized_img.size[0]}x{resized_img.size[1]}")

        # Save the processed image, overwriting the original
        resized_img.save(file_path)
        print(f"Processed {os.path.basename(file_path)} - cropped square around person and resized to 1024x1024")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("All images processed.") 
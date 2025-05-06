#!/usr/bin/env python3
import os
import glob

def process_file(file_path):
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace
    modified_content = content.replace("Photo of MARK", "a photo of ohwx man")
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)
    
    print(f"Processed: {file_path}")

def main():
    # Get all .txt files in the processed directory
    txt_files = glob.glob("./processed3/*.txt")
    
    if not txt_files:
        print("No .txt files found.")
        return
    
    # Process each text file
    for file_path in txt_files:
        process_file(file_path)
    
    print(f"Completed processing {len(txt_files)} files.")

if __name__ == "__main__":
    main() 
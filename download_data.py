#from kagglehub official
import kagglehub

# Download latest version
path = kagglehub.dataset_download("zara2099/fashion-style-image-dataset")

print("Path to dataset files:", path)
'''
#from gemini
import kagglehub
import os

# 1. Find this "handle" from your Kaggle dataset's URL
# Example: 'anandhuh/fashion-product-images-dataset'
DATASET_HANDLE = "zara2099/fashion-style-image-dataset"

# 2. Set the path to download to (current folder)
DOWNLOAD_PATH = '.'

# 3. Download the files
print(f"Attempting to download dataset: {DATASET_HANDLE}")

try:
    # This will download all files from that dataset
    kagglehub.dataset_download_files(
        DATASET_HANDLE,
        path=DOWNLOAD_PATH,
        quiet=False  # Set to True to hide download progress
    )
    
    print("\nDownload complete!")
    print(f"Please check your folder for the 'data.csv' file.")

except Exception as e:
    print(f"\nAn error occurred during download:")
    print(e)
    print("\nPlease ensure:")
    print("1. You have the correct dataset handle.")
    print("2. Your server is authenticated with Kaggle (e.g., with a kaggle.json file).")'''
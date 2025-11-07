#from kagglehub official
import kagglehub

# Download latest version
path = kagglehub.dataset_download("zara2099/fashion-style-image-dataset")

print("Path to dataset files:", path)

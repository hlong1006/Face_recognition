import kagglehub

# Download latest version
path = kagglehub.dataset_download("anthonytherrien/image-classification-64-classes-animal")

print("Path to dataset files:", path)
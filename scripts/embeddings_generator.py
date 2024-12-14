import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ResNet model setup
model = models.resnet101(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        embedding = model(input_tensor).flatten(start_dim=1)
    
    return embedding.cpu().numpy().flatten()

def generate_embeddings(image_paths):
    embeddings = []
    for image_path in image_paths:
        try:
            embedding = process_image(image_path)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    return embeddings

if __name__ == "__main__":
    pass
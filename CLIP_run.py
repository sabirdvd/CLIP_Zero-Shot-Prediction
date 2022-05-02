
import os
import glob
import sys
import torch
import torchvision.transforms as Transforms
import clip
from PIL import Image
import argparse
import re


parser=argparse.ArgumentParser()
parser.add_argument('--c',  default='imagenet_classes.txt', help='', type=str,required=True)  
parser.add_argument('--s', default='ViT-B/32', help='model size', type=str,required=True)  
parser.add_argument('--i', default='ViT-B/32', help='path to the image file', type=str,required=True)  
args = parser.parse_args()



# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = torch.device("cpu")
print(f"Device - {device}")

# Load CLIP model
clip_model, clip_preprocess = clip.load('ViT-B/32', device)
clip_model.eval()

#
with open(args.c, "r") as f:
    categories = [s.strip() for s in f.readlines()]

text = clip.tokenize(categories).to(device)

def predict_clip(image_file_path):
    image = clip_preprocess(Image.open(image_file_path)).unsqueeze(0).to(device)
    clip_model, _ = clip.load(args.s, device)

    # Calculate features
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    predictions = {}
    for value, index in zip(values, indices):
        predictions[f"{categories[index]:>16s}"] = f"{1 * value.item():.4f}%"
	
    return predictions

# run pred 
#filenames = glob.glob("/image/*.jpg")
filenames = glob.glob(args.i)
filenames.sort()
for image in filenames:
     print(os.path.basename(image), predict_clip(image))
#print(predict_clip("image.jpg"))




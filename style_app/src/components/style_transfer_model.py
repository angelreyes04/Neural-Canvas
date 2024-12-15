import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import time
from torchvision.models import vgg19, VGG19_Weights

def apply_style_transfer(content_image_path, style_descriptions, output_path, num_styles=3, do_prune_model=False, do_quantize_model=False):
    """Complete pipeline for style transfer"""
    # Get style image paths and weights
    style_paths = []
    style_weights = []
    for text_query in style_descriptions:
        temp_style_paths, temp_style_weights = get_k_most_similar_image_paths_and_weights(text_query, k=num_styles)
        style_paths += temp_style_paths
        style_weights += temp_style_weights

    print(f"Found {len(style_paths)} style images for '{text_query}'")
    print(style_weights)

    # Visualize style images
    for path in style_paths:
        plt.imshow(Image.open(path))
        plt.show()

    # Load and preprocess images
    content_image = load_and_preprocess_image(content_image_path)
    style_images = [load_and_preprocess_image(path) for path in style_paths]

    # Perform style transfer
    output = optimize_image(
        content_image=content_image,
        style_images=style_images,
        weights=style_weights,
        do_prune_model=do_prune_model,
        do_quantize_model=do_quantize_model
    )

    # Save output image
    save_output_image(output, output_path)

    print(f"Style transfer complete! Result saved to {output_path}")

    return output

    # Freeze remaining layers to prevent further changes
    for param in model.vgg.parameters():
        param.requires_grad = False

    return model

def get_k_most_similar_image_paths_and_weights(text, k):
    # Get text embedding
    text_embedding = get_text_embedding(text)

    # Compute similarity
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    all_embeddings_norm = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)
    similarities = torch.mm(text_embedding, all_embeddings_norm.T)

    # Get top-k indices and similarities
    top_k = torch.topk(similarities, k=k, dim=1)
    top_k_indices = top_k.indices.squeeze(0).tolist()
    top_k_similarities = top_k.values.squeeze(0).tolist()

    # Normalize similarities to get weights
    similarity_sum = sum(top_k_similarities)
    weights = [sim / similarity_sum for sim in top_k_similarities]

    top_k_image_paths = df.iloc[top_k_indices]['Image Path'].tolist()
    return top_k_image_paths, weights


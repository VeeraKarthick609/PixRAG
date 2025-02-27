# app.py
import streamlit as st
import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

# Import our custom modules
from src.models.vit import ViT
from src.models.retrieval import retrieve_similar_images
from src.models.generative import Generator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_vit_model():
    # For feature extraction, we load the fine-tuned ViT and remove the classification head
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=100,
                embed_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1)
    # Load fine-tuned weights if available
    try:
        model.load_state_dict(torch.load("checkpoints/vit_cifar100_best.pth", map_location=device))
        st.write("Loaded fine-tuned ViT model.")
    except Exception as e:
        st.write("Using randomly initialized ViT model.", e)
    model.to(device)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_generator():
    gen = Generator(latent_dim=100, condition_dim=128)
    gen.to(device)
    gen.eval()
    return gen

vit_model = load_vit_model()
generator = load_generator()

# Transformation for ViT
vit_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Our ViT is built for 32x32 images (CIFAR-100)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

st.title("ViT-RAG: Image Retrieval & Generation")
mode = st.sidebar.radio("Select Operation Mode:", ("Retrieval", "Generation"))

uploaded_file = st.file_uploader("Upload a query image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Query Image", use_column_width=True)
    
    # Preprocess for ViT
    input_image = vit_transform(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        # Extract features using ViT; here we take the penultimate layer (or use the classification token)
        # For simplicity, we forward and then ignore classification head by taking the embedding from the penultimate layer.
        # In our ViT, the forward returns logits, so you may consider modifying the model to output features.
        # Here we simply use the logits as a proxy.
        query_logits = vit_model(input_image)
    # For demonstration, we use the logits as the feature vector.
    query_embedding = query_logits.cpu().numpy().squeeze()
    
    if mode == "Retrieval":
        st.subheader("Retrieval Mode")
        st.write("Finding similar images...")
        # Use CIFAR-100 test set as dummy database
        db_dir = "data/cifar100"
        test_dataset = datasets.CIFAR100(root=db_dir, train=False, download=True, transform=transforms.ToTensor())
        num_db_images = len(test_dataset)
        embed_dim = query_embedding.shape[0]
        # For demonstration, simulate precomputed embeddings with random vectors
        np.random.seed(42)
        database_embeddings = np.random.rand(num_db_images, embed_dim)
        top_k = st.sidebar.number_input("Number of images to retrieve", min_value=1, max_value=20, value=5)
        indices, similarities = retrieve_similar_images(query_embedding, database_embeddings, top_k=top_k)
        st.write("Retrieved image indices and similarity scores:")
        for idx, sim in zip(indices, similarities):
            st.write(f"Index: {idx}, Similarity: {sim:.3f}")
        # Display retrieved images
        retrieved_images = [test_dataset[i][0] for i in indices]
        retrieved_images = [transforms.ToPILImage()(img) for img in retrieved_images]
        st.image(retrieved_images, caption=[f"Sim: {s:.2f}" for s in similarities], width=100)
        
    elif mode == "Generation":
        st.subheader("Generation Mode")
        st.write("Generating a synthetic image...")
        latent_dim = 100
        noise = torch.randn(1, latent_dim).to(device)
        with torch.no_grad():
            fake_image = generator(noise, query_logits)
        # Rescale image from tanh output (-1,1) to (0,1)
        fake_image = fake_image.squeeze().cpu()
        fake_image = (fake_image * 0.5 + 0.5).clamp(0, 1)
        fake_pil_image = transforms.ToPILImage()(fake_image)
        fake_pil_image = fake_pil_image.resize((128, 128))
        st.image(fake_pil_image, caption="Generated Image", use_column_width=False)
else:
    st.info("Please upload an image to start.")

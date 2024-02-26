from flask import Flask, request, jsonify
import torch
import json
import open_clip
import time
from PIL import Image
import torchvision.transforms.transforms as transforms
import io
import os

# Initialize the Flask application
app = Flask(__name__)

# Load configuration from environment variables
CONFIG = {
    "model": os.getenv("MODEL", "ViT-SO400M-14-SigLIP-384"),
    "model_name": os.getenv("MODEL_NAME", "siglip-so400m/14@384"),
    "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", 128)),
    "port": int(os.getenv("PORT", 1708)),
    "device": os.getenv("DEVICE", "cuda:0"),
    "debug": os.getenv("DEBUG", False)
}

print(CONFIG)

import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the device, model, and tokenizer
device = torch.device(CONFIG["device"])
logging.debug(f"Device set to {CONFIG['device']}")

try:
    pretrained_models = dict(open_clip.list_pretrained())
    model_name = CONFIG["model"]
    if model_name not in pretrained_models:
        raise ValueError(f"Model {model_name} is not available in pretrained models.")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, device=device, pretrained=pretrained_models[model_name], precision="fp16")
    model.eval()
    model.to(device).float()  # Move model to device and convert to half precision
    logging.debug(f"Model {model_name} loaded and moved to {device}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

try:
    tokenizer = open_clip.get_tokenizer(model_name)
    logging.debug(f"Tokenizer for model {model_name} obtained")
except Exception as e:
    logging.error(f"Error obtaining tokenizer for model {model_name}: {e}")
    raise

def get_image_embeddings(image_bytes):
    """
    Generate embeddings for an image in bytes.
    """
    logging.debug("Generating embeddings for the received image.")
    # Load and preprocess the image
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = preprocess(image).unsqueeze(0).to(device).float()  # Process image, then move and convert
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise

    # Generate embeddings
    with torch.no_grad():
        try:
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embeddings = image_features.cpu().numpy().tolist()
            logging.debug("Embeddings generated successfully.")
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            raise

    return embeddings

@app.route('/embeddings', methods=['POST'])
def embeddings():
    logging.debug("Received request for embeddings.")
    if 'image' not in request.files:
        logging.debug("Missing 'image' in files.")
        return jsonify({'error': 'missing file'}), 400
    file = request.files['image']
    if file:
        embeddings_response = {}
        logging.debug("Image file found, processing for embeddings.")
        image_bytes = file.read()

        # Start timing the embedding generation
        start_time = time.time()

        embeddings = get_image_embeddings(image_bytes)
        end_time = time.time()
        logging.debug(f"TURN ONE: Debug timing: Embedding generation took {end_time - start_time:.4f} seconds")

        embeddings_response['embeddings'] = embeddings
        embeddings_response['generation_time_sec'] = end_time - start_time

        logging.debug("Returning generated embeddings with timing information.")
        return jsonify(embeddings_response)
    else:
        logging.debug("Invalid or empty image file received.")
        return jsonify({'error': 'invalid file'}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=CONFIG['port'], debug=CONFIG['debug'])

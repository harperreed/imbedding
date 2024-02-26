
# OpenCLIP Flask Service with GPU Acceleration

This project provides a Flask-based web service for generating image embeddings using the OpenCLIP model, leveraging GPU acceleration for efficient processing. It's containerized with Docker, ensuring easy deployment and scalability.

## Features

- Flask web application for handling image embedding requests.
- Utilizes OpenCLIP pre-trained models with support for multiple configurations.
- GPU acceleration for high-performance computation of embeddings.
- Docker and Docker Compose integration for easy setup and deployment.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Docker and Docker Compose installed on your system.
- NVIDIA Docker for GPU support within Docker containers.
- An NVIDIA GPU with the appropriate drivers installed.

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Build and run the Docker container**

   Using Docker Compose, you can build and run the service with a single command:

   ```bash
   docker-compose up --build
   ```

   This command builds the Docker image and starts the service, making it accessible on port 1708.

## Usage

To use the service, send a POST request to the `/embeddings` endpoint with an image file. For example, using `curl`:

```bash
curl -F "image=@path_to_your_image.jpg" http://localhost:1708/embeddings
```

Replace `path_to_your_image.jpg` with the actual path to an image file. The service will return the image embeddings as a JSON response.

## Configuration

The service can be configured through environment variables in the `docker-compose.yml` file. Available configurations include:

- `MODEL`: OpenCLIP model variant (default: `ViT-SO400M-14-SigLIP-384`).
- `MODEL_NAME`: Model name for loading pre-trained weights.
- `MAX_BATCH_SIZE`: Maximum batch size for processing images.
- `PORT`: Port number the Flask application listens on.
- `DEVICE`: Device to run the model on (`cuda:0` for GPU).
- `DEBUG`: Enable/disable Flask debug mode.

## License

Distributed under the MIT License. See `LICENSE` for more information.

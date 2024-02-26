# Start from the PyTorch image that includes CUDA support
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if necessary)
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/*

# Copy the local code to the container
COPY . .

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 1708 available to the world outside this container
EXPOSE 1708

# Define environment variable (adjust as needed)
# ENV MODEL_NAME ViT-B/32

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=1708"]

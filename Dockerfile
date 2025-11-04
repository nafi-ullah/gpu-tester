# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY gpu_test_app.py .

# Expose port
EXPOSE 8000

# Start the FastAPI app using Uvicorn
CMD ["uvicorn", "gpu_test_app:app", "--host", "0.0.0.0", "--port", "8000"]

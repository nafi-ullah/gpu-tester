# Use NVIDIA’s official PyTorch CUDA image
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gpu_test_app.py .

# Automatically verify GPU availability at container startup
RUN python -c "import torch; assert torch.cuda.is_available(), 'GPU not accessible!'"

EXPOSE 8000

# Set NVIDIA runtime environment automatically
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uvicorn", "gpu_test_app:app", "--host", "0.0.0.0", "--port", "8000"]

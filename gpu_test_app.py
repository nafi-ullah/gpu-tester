from fastapi import FastAPI, BackgroundTasks
import torch
import time
import logging
from ultralytics import YOLO
import pynvml

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # first GPU

app = FastAPI(title="GPU Stress Test App")

def log_gpu_usage():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    logging.info(f"GPU Memory Usage: {mem_info.used / 1024**2:.2f} MB / {mem_info.total / 1024**2:.2f} MB")
    logging.info(f"GPU Utilization: {pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu}%")

def stress_test_gpu(duration_minutes: int = 30, batch_size: int = 16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load YOLOv8n (nano, small model) to fit 2-3GB easily
    model = YOLO("yolov8n.pt")  # YOLOv8n pretrained
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, 640, 640).to(device)

    end_time = time.time() + duration_minutes * 60
    iteration = 0

    while time.time() < end_time:
        iteration += 1
        logging.info(f"Iteration {iteration}: Running inference...")
        with torch.no_grad():
            outputs = model(dummy_input)
        log_gpu_usage()
        time.sleep(1)  # optional, to make logs readable

    logging.info("GPU stress test completed!")

@app.get("/start-test")
def start_test(background_tasks: BackgroundTasks):
    """
    Start a 30-minute GPU stress test in the background.
    """
    background_tasks.add_task(stress_test_gpu)
    return {"message": "GPU stress test started in background. Check logs for usage."}

@app.get("/gpu-status")
def gpu_status():
    """
    Check current GPU usage.
    """
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return {
        "gpu_memory_used_MB": mem_info.used / 1024**2,
        "gpu_memory_total_MB": mem_info.total / 1024**2,
        "gpu_utilization_percent": utilization.gpu
    }

import pandas as pd
from ultralytics import YOLO
# Load YOLO model
model = YOLO('yolov8s.pt')
model.data = r"D:\FINALGCSDP\Dataset\data.yaml"
print("Dataset configuration:", model.data) 
# Train the model
model.train(
    data= r"D:\FINALGCSDP\Dataset\data.yaml",  
    epochs=50,            
    imgsz=640,            
    batch=21,
    device = '0', # Using the GPU, If you get an error here, check how to use GPU with PyTorch
) 
# Save the retrained model
model.save(r"D:\FINALGCSDP\yolov8s_retrained_NEW.pt")  
print("Model saved successfully!")
# Start validation
print("Starting validation...")
metrics = model.val(
    data= r"D:\FINALGCSDP\Dataset\data.yaml",  
    batch=3,        
    imgsz=640              
) 
# Inspect metrics object
print("Metrics object content:")
print(metrics)
# Debug: Check available attributes in metrics
print("Available attributes in metrics:")
print(dir(metrics))
# Extract metrics
try:
    metrics_dict = {
        "precision": float(metrics.box[0]) if hasattr(metrics, 'box') else None,
        "recall": float(metrics.box[1]) if hasattr(metrics, 'box') else None,
        "mAP@50": float(metrics.box[2]) if hasattr(metrics, 'box') else None,
        "mAP@50-95": float(metrics.box[3]) if hasattr(metrics, 'box') else None
    }
except Exception as e:
    print(f"Error extracting metrics: {e}")
    metrics_dict = {
        "precision": None,
        "recall": None,
        "mAP@50": None,
        "mAP@50-95": None
    }
# Debug: Check extracted metrics
print("Extracted metrics:")
print(metrics_dict)
# Save metrics to CSV
csv_file_path = r"D:\FINALGCSDP\ValidationMetrics.csv"
try:
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    metrics_df.to_csv(csv_file_path, index=False)
    print(f"Validation metrics saved to {csv_file_path}")
except Exception as e:
    print(f"Error while saving metrics to CSV: {e}")


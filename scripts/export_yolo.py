from ultralytics import YOLO
import os
import shutil

def main():
    print("Loading YOLO11n model...")
    # This will automatically download yolo11n.pt if not present
    model = YOLO("yolo11n.pt") 

    print("Exporting model to TFLite (INT8)...")
    # Export to tflite, enabling INT8 quantization for edge deployment
    # Ultralytics automatically handles the representative dataset for basic quantization
    export_path = model.export(format="tflite", int8=True, imgsz=320)
    
    # export_path is usually the directory or the file where it was saved
    # typically it creates a dir named yolo11n_saved_model/ or simply yolo11n_int8.tflite
    # Let's locate the generated .tflite file.
    
    exported_file = "yolo11n_saved_model/yolo11n_int8.tflite"
    if not os.path.exists(exported_file):
        # Fallback to current directory if ultralytics does not put it in a subfolder
        exported_file = "yolo11n_int8.tflite"

    print(f"Exported file should be at: {export_path}")
    
if __name__ == "__main__":
    main()

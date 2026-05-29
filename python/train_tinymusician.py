import torch
import torch.nn as nn
import os

class TinyMusician(nn.Module):
    """
    TinyMusician Model
    Takes 4 scene heuristic parameters (density, valence, arousal, timbre) 
    and outputs 20 parameters:
    - [0:4]: 4 "Tweaker" offsets to nudge the deterministic parameters.
    - [4:20]: 16 simulated "Composer" notes (e.g., scale degree choices or MIDI offsets).
    """
    def __init__(self):
        super(TinyMusician, self).__init__()
        # Very simple lightweight network for edge devices
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 20)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    print("Generating TinyMusician ONNX training artifacts...")
    model = TinyMusician()
    
    # Enable training mode and gradients for all parameters
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # 1. Export base ONNX model
    dummy_input = torch.randn(1, 4)
    base_model_path = "tinymusician_base.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        base_model_path, 
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Exported base ONNX model to {base_model_path}")

    # 2. Generate training artifacts for On-Device Training (ODT)
    artifact_dir = "artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    try:
        from onnxruntime.training import artifacts
        requires = artifacts.Requires(
            [artifacts.Requires.grad("fc1.weight"), 
             artifacts.Requires.grad("fc1.bias"), 
             artifacts.Requires.grad("fc2.weight"), 
             artifacts.Requires.grad("fc2.bias")]
        )
        artifacts.generate_artifacts(
            base_model_path,
            requires=requires,
            loss=artifacts.Loss(
                artifacts.LossType.MSELoss, 
                loss_name="loss", 
                target_name="target"
            ),
            optimizer=artifacts.Optimizer(
                artifacts.OptimType.AdamW, 
                learning_rate=1e-3
            ),
            artifact_directory=artifact_dir
        )
        print(f"Training artifacts generated in './{artifact_dir}/'")
    except ImportError:
        print("onnxruntime.training not found. Creating mock artifacts for testing.")
        import shutil
        shutil.copy(base_model_path, os.path.join(artifact_dir, "training_model.onnx"))
        open(os.path.join(artifact_dir, "eval_model.onnx"), "w").close()
        open(os.path.join(artifact_dir, "optimizer_model.onnx"), "w").close()
        open(os.path.join(artifact_dir, "checkpoint"), "w").close()
        print(f"Mock training artifacts generated in './{artifact_dir}/'")

    print("Move these files to the Android app's assets directory to begin Phase 5 ODT.")

if __name__ == "__main__":
    main()

import os
import torch

def load_trained_model(weights_path, inital_model, device = torch.device('cuda')):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"File not found: {weights_path}")
    model_weights = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = model_weights if isinstance(model_weights, dict) else model_weights.state_dict()
    inital_model.load_state_dict(state_dict)
    inital_model.to(device)
    inital_model.eval()
    print(f"Loaded model weights from {weights_path}")
    return inital_model
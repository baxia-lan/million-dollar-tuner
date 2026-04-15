import torch
import sys
try:
    # First try weights_only=True (safe mode)
    ckpt = torch.load("/Users/sarsa/claude/Applio/rvc/models/predictors/rmvpe.pt", map_location="cpu", weights_only=True)
    print("Loaded with weights_only=True OK")
    print(type(ckpt))
except Exception as e:
    print(f"weights_only=True failed: {e}")
    try:
        # Try weights_only=False (allows pickle)
        ckpt = torch.load("/Users/sarsa/claude/Applio/rvc/models/predictors/rmvpe.pt", map_location="cpu", weights_only=False)
        print("Loaded with weights_only=False OK")
        print(type(ckpt))
    except Exception as e2:
        print(f"weights_only=False also failed: {e2}")
        print("File is likely corrupted. Need to re-download.")

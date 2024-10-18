import torch
import numpy as np

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

def numpy_to_list(ar):
    res = []
    for row in ar.tolist():
        res += row
    return res

def list_arrays_to_list(np_list):
    features = []
    for ar in np_list:
        features += numpy_to_list(ar)
    return features

class State:
    "Put all features as a list"
    def __init__(self, features: list = []):
        self.features = features

    def to_tensor(self):
        feature_list = []
        for feature in self.features:
            if isinstance(feature, np.ndarray):
                feature_list.append(feature.flatten())
            elif isinstance(feature, list):
                feature_list.append(np.array(feature))
            elif isinstance(feature, int) or isinstance(feature, float) or isinstance(feature, bool):
                feature_list.append(np.array([feature], dtype=np.float32))
        return torch.FloatTensor(np.concatenate(feature_list)).to(DEVICE)
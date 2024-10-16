import torch
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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
            l = feature
            if isinstance(feature[0], np.ndarray):
                l = list_arrays_to_list(feature)
            elif not isinstance(feature, list):
                raise(f"Passed non list element to features: {feature}")
            feature_list += l
        return torch.tensor(feature_list, dtype=torch.float32).squeeze(0).to(device)
import torch


def from_numpy(data):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    data = torch.from_numpy(data)
    if data.dtype == torch.float64:
        data = data.float()
    return data.to(device)

def to_numpy(tensor):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    return tensor.to("cpu").detach().numpy()
    
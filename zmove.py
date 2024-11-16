import os
import torch

for folder in [
    'tensors/globals/23204fccf8640f80d63a5150354b927e/input',
    'tensors/locals/fcc27ab0a9a9f59d230ba37ada8b0da4/input',
    'tensors/locals/fcc27ab0a9a9f59d230ba37ada8b0da4/output',
]:
    filenames = os.listdir(folder)
    for filename in filenames:
        if filename.endswith(".pt"):
            path = os.path.join(folder, filename)
            tensor = torch.load(path)
            if not tensor.is_cuda:
                tensor_gpu = tensor.cuda()
                torch.save(tensor_gpu, path)
                print(f"Transferred {filename} to GPU and saved to {path}")
            else:
                print(f"{filename} is already on GPU.")

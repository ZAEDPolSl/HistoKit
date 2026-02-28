# python
# python
import torch

def list_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            devices.append(f"cuda:{i} - {name} (SM {cap[0]}.{cap[1]}, {mem_total} GiB)")
    return devices

if __name__ == "__main__":
    for d in list_devices():
        print(d)

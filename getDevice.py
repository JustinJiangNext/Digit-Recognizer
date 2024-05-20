import torch


def getDevice(deviceName = "Default"):
    device:torch.device = None
    if deviceName == "Default":
        if (torch.backends.mps.is_available()):
            device = torch.device('mps')
        elif(torch.cuda.is_available()):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(deviceName)
    
    return device
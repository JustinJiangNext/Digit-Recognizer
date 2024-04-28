import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class loadData:
    device:torch.device = None
    trainingLoader:torch.utils.data.DataLoader = None
    testData:torch.utils.data.DataLoader = None

    inputSize:int = 28 * 28 #Image Size
    batchSize :int = 100 #Number of images per batch

    def __init__(self, deviceName = "Default"): 
        if deviceName == "Default":
            if (torch.backends.mps.is_available()): 
                self.device = torch.device('mps')
            elif(torch.cuda.is_available()):
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(deviceName)

        # MNIST dataset 
        trainingData:torch.datasets.MNIST = torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                                transform=transforms.ToTensor(),  
                                                download=True)

        testingData:torch.datasets.MNIST = torchvision.datasets.MNIST(root='./data', 
                                                train=False, 
                                                transform=transforms.ToTensor())

        self.trainingLoader = torch.utils.data.DataLoader(dataset=trainingData, 
                                                batch_size=self.batchSize, 
                                                shuffle=True)

        self.testData = torch.utils.data.DataLoader(dataset=testingData, 
                                                batch_size=self.batchSize, 
                                                shuffle=False)
        
    def getTrainingData(self) -> tuple:
        trainingImages:torch.tensor = None
        trainingImagesLabels:torch.tensor = None

        for i, (images, labels) in enumerate(self.trainingLoader):  
            """
            Each batch contains 100 images in a 100 x 28 x 28 tensor
            Convert to 100 x 784 tensor, then merge with rest of tensors
            For labels, merge 100 x 1 with rest, no resizing first
            """
            if i == 0:
                trainingImages = images.reshape(-1, self.inputSize).to(self.device) 
                trainingImagesLabels = labels.to(self.device)
            else:
                torch.cat(trainingImages, images.reshape(-1, self.inputSize).to(self.device))
                torch.cat(trainingImagesLabels, labels.to(self.device))
        return (trainingImages, trainingImagesLabels)
    
    def getTrainingData(self) -> tuple:
        testingImages:torch.tensor = None
        testingImagesLabels:torch.tensor = None

        for i, (images, labels) in enumerate(self.testingLoader):  
            """
            Each batch contains 100 images in a 100 x 28 x 28 tensor
            Convert to 100 x 784 tensor, then merge with rest of tensors
            For labels, merge 100 x 1 with rest, no resizing first
            """
            if i == 0:
                testingImages = images.reshape(-1, self.inputSize).to(self.device) 
                testingImagesLabels = labels.to(self.device)
            else:
                torch.cat(testingImages, images.reshape(-1, self.inputSize).to(self.device))
                torch.cat(testingImagesLabels, labels.to(self.device))
        return (testingImages, testingImagesLabels)

    

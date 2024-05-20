import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from getDevice import getDevice 


class loadData:
    def __init__(self, deviceName = "Default"): 
        self.device:torch.device = None
        self.trainingLoader:torch.utils.data.DataLoader = None
        self.testData:torch.utils.data.DataLoader = None

        self.inputSize:int = 28 * 28 #Image Size
        self.device = getDevice(deviceName)

        # MNIST dataset 
        trainingData:torch.datasets.MNIST = torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                                transform=transforms.ToTensor(),  
                                                download=True)

        testingData:torch.datasets.MNIST = torchvision.datasets.MNIST(root='./data', 
                                                train=False, 
                                                transform=transforms.ToTensor())

        self.trainingDataSize = len(trainingData)
        self.testingDataSize = len(testingData)

        trainingLoader = torch.utils.data.DataLoader(dataset=trainingData, batch_size = self.trainingDataSize)
        testDataLoader = torch.utils.data.DataLoader(dataset=testingData, batch_size = self.testingDataSize)

        


        self.trainImage, self.trainLabel = next(iter(trainingLoader))
        self.trainImage = self.trainImage.to(self.device)
        self.trainLabel = self.trainLabel.to(self.device)

        self.testImage, self.testLabel = next(iter(testDataLoader))
        self.testImage = self.testImage.to(self.device)
        self.testLabel = self.testLabel.to(self.device)



        
    def getTrainingData(self) -> tuple:
        return self.trainImage, self.trainLabel
    
    def getTestingData(self) -> tuple:
        return self.testImage, self.testLabel

    
    def getTrainingSize(self) -> int:
        return self.trainingDataSize
    
    def getTestingSize(self) -> int:
        return self.testingDataSize

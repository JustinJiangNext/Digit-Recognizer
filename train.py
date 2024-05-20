import torch 
import torch.nn as nn
from getDevice import getDevice 
from cnnMNIST import cnnMNIST
from loadData import loadData

dataLoader = loadData()

def train(epochs:int, learning_rate:float, logging:bool = False):
    device = getDevice()
    model = cnnMNIST().to(device)
    #image, label = dataLoader.getTrainingData()
    image = dataLoader.trainImage
    label = dataLoader.trainLabel
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if logging and epoch % 10 == 0:
                print(epoch)
    
    return model

def test(model):
    numCorrect = 0
    samples = dataLoader.getTestingSize()
    images, labels = dataLoader.getTestingData()
    output = model(images)
    _, predicted = torch.max(output, 1)
    numCorrect = (predicted == labels).sum().item()
    print(numCorrect)
    print(samples)
    print(images.size())
    print("accuracy = " + str((float(numCorrect)/samples)))



test(train(150, 0.01, logging=True))

'''Building a CNN model for digit classification using pytorch'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchsummary

from model import Model
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import mnist
train_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

# examples = iter(train_loader)
# features, label = next(examples)
# print(features.shape)
# print(label.shape)

# Training loop + Testing
def main(reset_weights: bool = True):
    model = Model().to(device)

    # Reset weights first
    if reset_weights:
        model.apply(fn=model.weight_reset)
        print('Model weights successfully reset')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)

    n_total_steps = len(train_loader)

    # Training Loop
    for epoch in range(config.epochs):
        for i, (images, labels) in enumerate(train_loader):
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                print (f'Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Finished Training')
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

    #Evaluation
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            n_correct += (preds == labels).sum().item()
            n_samples += len(labels)

    print(f'Accuracy: {round(100*(n_correct / n_samples), 3)}%')
    
if __name__ == "__main__":
    main()








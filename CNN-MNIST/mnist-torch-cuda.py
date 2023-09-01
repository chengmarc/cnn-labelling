# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
try:
    # Import core libraries
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision.transforms as transforms

    # Import libraries for utilities
    import os, time, random, subprocess
    import matplotlib.pyplot as plt
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from colorama import init, Fore
    init()
    print(Fore.GREEN + "All libraries imported.")

except:
    print("Dependencies missing, please use pip to install all dependencies:")
    print("torch, torchvision, os, time, random, subprocess, matplotlib, PIL, colorama")
    input('Press any key to quit.')
    exit()

# %% Checking if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(Fore.GREEN + "Detected GPU, connected successfully to CUDA.")
else:
    device = torch.device("cpu")
    print(Fore.YELLOW + "Caution: no GPU detected.")
    print(Fore.YELLOW + "This issue may be due to the fact that your computer does not have an Nvidia GPU or does not have the correct version of CUDA.")
    print(Fore.YELLOW + "Consequence: torch will use CPU for computation.")
    print(Fore.YELLOW + "Consequence: model training is expected to be slow.")

# %% Load MNIST data into system memory
directory = os.path.dirname(os.path.realpath(__file__))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

dataset_train = MNIST(root=directory, train=True, transform=transform, download=True)
dataset_test = MNIST(root=directory, train=False, transform=transform, download=True)

loader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
loader_test = DataLoader(dataset=dataset_test, batch_size=64, shuffle=True)

print(Fore.WHITE + "MNIST dataset loaded.")

# %% Visualization of random samples
visualize = True
if visualize:
    index, fig = 0, plt.figure(figsize=(5, 5))
    for i in [random.randint(0, len(dataset_train)-1) for j in range(25)]:
        index += 1
        fig.add_subplot(5, 5, index)
        plt.xticks([])
        plt.yticks([])
        
        image_slice = dataset_train[i][0].numpy()
        plt.imshow(image_slice[0])            
    print(Fore.WHITE + "Visualizing 25 random samples, close the image to continue...")
    plt.show()    
del visualize, i, index, fig, image_slice, dataset_test, dataset_train

# %% Initialize CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print(Fore.WHITE + "CNN Model Initialized.")

# %% Model training
def train_model(loader, model) -> nn.Module:
    start_time = time.time()
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            scores = model(inputs)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()      
            if i % 100 == 99:
                average_loss = running_loss / 100
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {average_loss:.3f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(Fore.GREEN + f"Total training time: {total_time:.2f} seconds")
    return model

#Training Time Testing
#Epochs             - 5
#Batchsize          - 64
#Batches            - 938
#GPU: GTX-1060 6GB  - about 70 seconds
#GPU: RTX-3060 12GB - about 40 seconds

# %% Main Execution
if not os.path.isfile(directory + '/pre_trained_model.pt'):
    print(Fore.WHITE + "No pre-trained model detected.")
    print(Fore.WHITE + "Start training...")
    model = train_model(loader_train, model)       
    torch.save(model.state_dict(), directory + '/pre_trained_model.pt')
    print(Fore.GREEN + "Model have been saved to the default location.")
else:
    print(Fore.GREEN + "Pre-trained model detected.")
    model.load_state_dict(torch.load(directory + '/pre_trained_model.pt'))
    print(Fore.GREEN + "Pre-trained model loaded.")
    
# %% Model testing
def check_accuracy(loader, model) -> float:
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = f"{num_correct/num_samples * 100:.2f}%"
    print(Fore.WHITE + "This model has a", accuracy, "accuracy on the test dataset.")
    return accuracy

check_accuracy(loader_test, model)

# %% Model prediction
boolean = 'Y'
print(Fore.WHITE + "Please write a character in MS Paint and save it.")

while boolean == 'Y' or boolean == 'y':

    image = Image.new('L', (28, 28), color=0)
    image_path = 'test_image.png'
    image.save(image_path)

    process = subprocess.Popen(['mspaint', image_path])
    process.wait()

    modified_image = Image.open(image_path)
    modified_image = modified_image.convert('L')
    modified_image = transform(modified_image).to(device)
    modified_image = modified_image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        output = model(modified_image)

    softmax_scores = F.softmax(output, dim=1)
    softmax_scores = softmax_scores.tolist()[0]
    softmax_scores = ["{:0>5.2f}%".format(score*100) for score in softmax_scores]
    
    print(Fore.WHITE + "Predicted character:")
    for i in range(0, 10):
        print(Fore.WHITE, i, ":", softmax_scores[i])
    print("")
    del i, boolean, image_path, image, modified_image
    boolean = input(Fore.WHITE + "Do you want to write another character? (Y/N)")

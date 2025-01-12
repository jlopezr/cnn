import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# Transformaciones para normalizar y convertir a tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar el dataset MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Definir la arquitectura de la red neuronal clásica
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# Definir la arquitectura de la red neuronal convolucional
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# Función para entrenar el modelo
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tiempo total de entrenamiento: {total_time:.2f} segundos")
    return history

# Función para evaluar el modelo
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(testloader)
    val_accuracy = 100 * correct / total
    print(f'Precisión en el conjunto de prueba: {val_accuracy:.2f}%, Pérdida: {val_loss}')
    return val_loss, val_accuracy

# Entrenar y evaluar la red neuronal clásica
model_nn = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)
history_nn = train_model(model_nn, trainloader, criterion, optimizer)
val_loss_nn, val_accuracy_nn = evaluate_model(model_nn, testloader)
history_nn['val_loss'].append(val_loss_nn)
history_nn['val_accuracy'].append(val_accuracy_nn)

# Entrenar y evaluar la red neuronal convolucional
model_cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)
history_cnn = train_model(model_cnn, trainloader, criterion, optimizer)
val_loss_cnn, val_accuracy_cnn = evaluate_model(model_cnn, testloader)
history_cnn['val_loss'].append(val_loss_cnn)
history_cnn['val_accuracy'].append(val_accuracy_cnn)

# Comparar resultados
plt.plot(history_nn['accuracy'], label='NN - Entrenamiento')
plt.plot(history_nn['val_accuracy'], label='NN - Validación')
plt.plot(history_cnn['accuracy'], label='CNN - Entrenamiento')
plt.plot(history_cnn['val_accuracy'], label='CNN - Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

plt.plot(history_nn['loss'], label='NN - Entrenamiento')
plt.plot(history_nn['val_loss'], label='NN - Validación')
plt.plot(history_cnn['loss'], label='CNN - Entrenamiento')
plt.plot(history_cnn['val_loss'], label='CNN - Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
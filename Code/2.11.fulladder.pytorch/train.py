import torch
from torchvision import transforms, utils, datasets
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from model import MLP
import time
from datagen import generate_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

x, y = generate_data()
train_x = torch.Tensor(x[:400])
train_y = torch.Tensor(y[:400])
test_x = torch.Tensor(x[400:])
test_y = torch.Tensor(y[400:])

epochs = 200
model = MLP(input_dim=9, hidden_1=20, hidden_2=20, output_dim=5)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_acc = 0.0

for epoch in range(epochs):
    # Train model
    model.train()
    running_loss = 0.0
    t = time.perf_counter()
    for index, _ in enumerate(train_x):
        data = train_x[index]
        result = train_y[index]
        outputs = model(data)
        optimizer.zero_grad()
        loss = loss_function(outputs, result)
        running_loss += loss
        loss.backward()
        optimizer.step()

        rate = index / len(train_x)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(
            "\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100)+1, a, b, loss), end="")
    print("\n time.perf_counter()-t1")

    model.eval()
    acc = 0.0
    with torch.no_grad():
        for index, _ in enumerate(test_x):
            data = test_x[index]
            result = test_y[index]
            outputs = model(data)
            for index, eachx in enumerate(outputs):
                if eachx > 0.5:
                    outputs[index] = 1
                else:
                    outputs[index] = 0
            if 0 == ((result != outputs).sum()):
                acc+=1
        acc = acc / len(test_x)
        if acc > best_acc:
            best_acc = acc
            print("Saving Model")
            torch.save(model.state_dict(), 'MLP_weights.pth')
            torch.save(model, 'MLP.pth')
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss, acc))

print('Finished Training')
import numpy as np
import torch


def train(model, X, y, epoch, optimizer, criterion, print_rate, D, H, W, iso_val=0.0):
    model.cuda()
    X.cuda()
    y.cuda()
    model.train()
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y)
        if (epoch % print_rate == 0):
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            if (max(y_pred) < iso_val or min(y_pred) > iso_val):
                print("Not valid isosurface: min-max->", min(y_pred), max(y_pred))
        # Backward pass
        loss.backward()
        optimizer.step()
    return model


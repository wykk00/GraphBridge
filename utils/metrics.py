import torch
import numpy as np

def accuracy(preds, labels):
    preds = np.argmax(preds, axis=-1)
    acc = (preds == labels).sum() / len(labels)
    return acc
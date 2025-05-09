import matplotlib.pyplot as plt
import numpy as np

def plot(train, test, title, labeltrain, labeltest):
    fig, ax = plt.subplots(figsize=(15, 5))

    plt.plot(np.arange(len(train)), train, label=labeltrain, linewidth=2)
    plt.plot(np.arange(len(test)), test, label=labeltest, linewidth=2)
    plt.title(title)
    plt.legend()
    plt.savefig(f"plots/{title}.png")
    plt.close()
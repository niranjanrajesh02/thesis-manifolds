import os
from dotenv import load_dotenv
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from data import index_to_label

load_dotenv()
save_path = os.getenv("PICKLE_DATA_PATH")
plot_path = os.getenv("PLOTS_PATH")
def plot_class_accs():
    with open(save_path + 'r50_class_accuracies.pkl', 'rb') as f:
        class_accs = pickle.load(f)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Sort the class accuracies by value descending
    class_accs = dict(sorted(class_accs.items(), key=lambda item: item[1], reverse=True))
    print(class_accs)
    x_vals = [index_to_label(int(cid))for cid in class_accs.keys()]
    y_vals = [acc for acc in class_accs.values()]

    sns.barplot(x=x_vals, y=y_vals, palette="viridis")
    plt.xticks(rotation=30)
    plt.ylabel('Top-1 Accuracy')
    plt.title('ResNet50 Top-1 Accuracy per Class')

    # plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'r50_class_accuracies.png'))
    



if __name__ == '__main__':
    plot_class_accs()
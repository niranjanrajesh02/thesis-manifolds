import os
from dotenv import load_dotenv
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from data import index_to_label
import pandas as pd

load_dotenv()
save_path = os.getenv("PICKLE_DATA_PATH")
plot_path = os.getenv("PLOTS_PATH")

def plot_class_accs(data='valid'):
    with open(save_path + f'r50_class_{data}_accuracies.pkl', 'rb') as f:
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
    
def plot_class_accdiff_manifold_dim():
    with open(save_path + f'r50_class_train_accuracies.pkl', 'rb') as f:
        train_class_accs = pickle.load(f)
    with open(save_path + f'r50_class_valid_accuracies.pkl', 'rb') as f:
        valid_class_accs = pickle.load(f)
    with open(save_path + f'r50_class_manifold_dims.pkl', 'rb') as f:
        class_man_dims = pickle.load(f)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 10))

    # Sort the class accuracies by name
    train_class_accs = dict(sorted(train_class_accs.items(), key=lambda item: item[0]))
    valid_class_accs = dict(sorted(valid_class_accs.items(), key=lambda item: item[0]))
    class_man_dims = dict(sorted(class_man_dims.items(), key=lambda item: item[0]))



    # checking that order is the same 
    assert train_class_accs.keys() == valid_class_accs.keys() == class_man_dims.keys()

    # Calculate the difference between train and valid accuracies
    class_diffs = {k: train_class_accs[k] - valid_class_accs[k] for k in train_class_accs.keys()}

    class_labels = [index_to_label(int(cid)) for cid in class_diffs.keys()]
    # make a df with the class, manifold dim, and diff and sort it based on diff
    df = pd.DataFrame({'class': class_labels, 'manifold_dim': list(class_man_dims.values()), 'diff': list(class_diffs.values())})
    df = df.sort_values(by='diff', ascending=True)
    
    # add a column for the x axis labels
    df['x_labels'] = df['class'] + '\n' + df['diff'].map('{:.2f}%'.format)
    
    sns.barplot(data=df, x='x_labels', y='manifold_dim', hue='class')
    # bar plot x axis labels
    plt.ylabel('Manifold Dimension')
    plt.xlabel('Classes')
    plt.title('ResNet50 Train-Valid Accuracy Difference per Class')
    plt.savefig(os.path.join(plot_path, 'r50_class_accdiff_manifold_dim.png'))


def plot_class_acc_manifold_dim(data='valid'):
    with open(save_path + f'r50_class_{data}_accuracies.pkl', 'rb') as f:
        class_accs = pickle.load(f)
    with open(save_path + f'r50_class_manifold_dims.pkl', 'rb') as f:
        class_man_dims = pickle.load(f)
      
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Sort the class accuracies by name
    class_accs = dict(sorted(class_accs.items(), key=lambda item: item[0]))
    class_man_dims = dict(sorted(class_man_dims.items(), key=lambda item: item[0]))
    assert class_accs.keys() == class_man_dims.keys()

    plot_dict = {k: (acc, class_man_dims[k]) for k, acc in class_accs.items()}
    plot_dict = dict(sorted(plot_dict.items(), key=lambda item: item[1][0], reverse=True))

    x_vals = [f'{index_to_label(int(cid))}\n{plot_dict[cid][0]:.2f}%' for cid in plot_dict.keys()]
    y_vals = [v[1] for v in plot_dict.values()]

    sns.barplot(x=x_vals, y=y_vals, palette="viridis")
    plt.ylabel('Manifold Dimension')
    plt.xlabel('Classes and their Top-1 Accuracy')
    plt.title(f'ResNet50 Top-1 {data.capitalize()} Accuracy and Manifold Dimension per Class')
    plt.savefig(os.path.join(plot_path, f'r50_class_acc_manifold_dim_{data}.png'))
    return

def plot_adv_acc():
    with open(save_path + 'r50_adv_accuracies.pkl', 'rb') as f:
        adv_accs = pickle.load(f)
    print(adv_accs)

if __name__ == '__main__':
    # plot_class_accs()
    # plot_class_accdiff_manifold_dim()
    # plot_class_acc_manifold_dim('train')
    plot_adv_acc()
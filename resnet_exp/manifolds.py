import pickle
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from data import index_to_label

load_dotenv()
data_path = os.getenv("PICKLE_DATA_PATH")
plot_path = os.getenv("PLOTS_PATH")

def estimate_linear_dim_PCA(activations, threshold=0.95):
    pca = PCA()
    pca.fit(activations)
    # Find the number of components required to explain 95% of the variance
    var = np.cumsum(pca.explained_variance_ratio_)
    # print("Cumulative variance: ", var)
    linear_dim = np.argmax(var > threshold) + 1
    # print("Linear dimension: ", linear_dim)
    return linear_dim

if __name__ == '__main__':
    class_edims = {}
    # Load the activations
    with open(os.path.join(data_path, 'r50_class_activations.pkl'), 'rb') as f:
        activations = pickle.load(f)
    for label, acts in activations.items():
        class_edim = estimate_linear_dim_PCA(acts, 0.95)
        class_edims[label] = class_edim
    # load classwise accuracies
    with open(os.path.join(data_path, 'r50_class_accuracies.pkl'), 'rb') as f:
        class_accuracies = pickle.load(f)
    print(f'class edims: {class_edims}')
    print(f'class accuracies: {class_accuracies}')
    assert class_edims.keys() == class_accuracies.keys()
    class_dict = {cid: [class_edims[cid], class_accuracies[cid]] for cid in class_edims.keys()}
    # sort the class_edims by the class_accuracies
    class_dict = dict(sorted(class_dict.items(), key=lambda x: x[1][1], reverse=True))
    print(f'class dict: {class_dict}')
    # plot class_edims as a barplot with sns and save it
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 8))
    x_vals = list(class_dict.keys())
    # x labels should be the class name and the class accuracy
    x_vals = [f'{index_to_label(int(x))}\n{class_dict[x][1]:.2f}%' for x in x_vals]
    # y labels should be the class edim
    y_vals = [class_dict[x][0] for x in class_dict.keys()]
    sns.barplot(x=x_vals, y=y_vals)

    # make axis titles smaller
    plt.xlabel('ImageNet Class and Top-1 Accuracy', fontsize=15)
    plt.ylabel('Estimated Dimension of Manifold by PCA', fontsize=15)
    # make axis labels smaller and have a tilt
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    


    plt.title('Estimated Dimension of Class Activations Manifolds for ResNet50', fontsize=30)
    plt.savefig(f'{plot_path}/r50_class_woof_edims.png')



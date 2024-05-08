import os
from dotenv import load_dotenv
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from data import index_to_label
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from my_utils import model_namer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


load_dotenv()
save_path = os.getenv("PICKLE_DATA_PATH")
plot_path = os.getenv("PLOTS_PATH")

def plot_class_accs(data='valid'):
    with open(save_path + f'r50_class_{data}_accuracies.pkl', 'rb') as f:
        class_accs = pickle.load(f)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 12))
    
    # Sort the class accuracies by value descending
    class_accs = dict(sorted(class_accs.items(), key=lambda item: item[1], reverse=True))
    print(class_accs)
    x_vals = [index_to_label(int(cid))for cid in class_accs.keys()]
    # if any x_val is repeated, add 2 to one of the repeated labels
    if 'terrier' in x_vals:
        x_vals[x_vals.index('terrier')] = 'terrier 0'
    y_vals = [acc for acc in class_accs.values()]

    sns.barplot(x=x_vals, y=y_vals, palette="viridis")
    plt.xticks(rotation=60)
    plt.ylabel('Top-1 Accuracy', fontsize=14)
    plt.title('ResNet50 Top-1 Accuracy per Class')
    # change font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

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
    
    sns.barplot(data=df, x='x_labels', y='manifold_dim')
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

def plot_adv_accs(model='ResNet50', n_classes=10):

    ext = f'_{n_classes}c' if n_classes != 10 else ''
    model = model.lower()[:3]
    print(model, ext)
    with open(save_path + f'{model}_adv_accuracies{ext}.pkl', 'rb') as f:
        adv_accs = pickle.load(f)
    print(adv_accs)

    # Sort the class accuracies by class name
    adv_accs = dict(sorted(adv_accs.items(), key=lambda item: item[0]))

    class_labels = [index_to_label(int(cid)) for cid in adv_accs.keys()]
    plot_df = pd.DataFrame({'class': class_labels, 'adv_acc': list(adv_accs.values())})

    # separate the adv_accs into columns based on epsilon
    i = 1
    epsilons = [0, 0.0005, 0.005]
    eps = epsilons[i]

    plot_df[f'adv_acc_{eps}'] = plot_df['adv_acc'].apply(lambda x: x[i])
    print(plot_df)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_df, x='class', y=f'adv_acc_{eps}')
    # turn off xticks
    plt.xticks([])

    plt.ylabel('Adversarial Accuracy (%)')
    plt.xlabel('Classes')
    plt.title(f'{model} Adversarial Accuracy per Class for eps={eps}')

    plt.savefig(os.path.join(plot_path, f'{model}_adv_accs{ext}.png'))
    return

def plot_adv_acc_manifold_dim(n_classes=10, all_eps=False):
    ext = f'_{n_classes}' if n_classes != 10 else ''

    with open(save_path + f'r50_adv_accuracies{ext}c.pkl', 'rb') as f:
        adv_accs = pickle.load(f)
    print(adv_accs)

    with open(save_path + f'r50_class_manifold_dims{ext}.pkl', 'rb') as f:
        cam_dims = pickle.load(f)
    
    # Sort the class accuracies by class name
    adv_accs = dict(sorted(adv_accs.items(), key=lambda item: item[0]))
    cam_dims = dict(sorted(cam_dims.items(), key=lambda item: item[0]))

    assert adv_accs.keys() == cam_dims.keys()

    class_labels = [index_to_label(int(cid)) for cid in adv_accs.keys()]
    plot_df = pd.DataFrame({'class': class_labels, 'adv_acc': list(adv_accs.values()), 'cam_dim': list(cam_dims.values())})

    epsilons = [0,0.0005, 0.005]
    # split adv_acc into 4 columns and name them based on the epsilon
    for i, eps in enumerate(epsilons):
        plot_df[f'adv_acc_{eps}'] = plot_df['adv_acc'].apply(lambda x: x[i])
    plot_df = plot_df.drop(columns=['adv_acc'])

    print(plot_df.keys())

    if all_eps:
        for eps in epsilons:
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=plot_df, x=f'adv_acc_{eps}', y='cam_dim', hue='class')
            # legend off
            plt.legend([],[], frameon=False)
            plt.ylabel('CAM Dimension')
            plt.xlabel('Adversarial Accuracy (%)')
            plt.title(f'ResNet50 Adversarial Accuracy and CAM Dimension per Class for eps={eps}')
            plt.savefig(os.path.join(plot_path, f'r50_adv_acc_cam_dim{ext}c_{eps}.png'))
    
    else:
        sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})
        # plt.figure(figsize=(12, 8))
        eps = 0.005     
        # round to nearest 5
        round_to_nearest_5 = lambda x: round(x / 5) * 5

        # round to nearest 1
        round_to_nearest_1 = lambda x: round(x)

    

        plot_df['rounded_acc'] = plot_df[f'adv_acc_{str(eps)}'].apply(round_to_nearest_5)
        
        # average the cam dims for all rows with same rounded acc
        plot_df = plot_df.groupby('rounded_acc')['cam_dim'].mean().reset_index()
        print(plot_df)

        rel = sns.scatterplot(data=plot_df, x=f'rounded_acc', y='cam_dim')

        reg = sns.regplot(data=plot_df, x=f'rounded_acc', y='cam_dim', scatter=False, color='blue', ax=rel)
        # legend off
        plt.legend([],[], frameon=False)

        # print the pearson correlation coefficient
        corr_coeff, p = pearsonr(plot_df[f'rounded_acc'], plot_df['cam_dim'])
        plt.suptitle(f'Pearson: {corr_coeff:.3f} p={float(p):.2e}', y=0.9, fontsize=14)

        plt.ylabel('CAM Dimension', fontsize=14)
        plt.xlabel('Adversarial Accuracy (%)', fontsize=14)
        
        plt.title("Adversarial Accuracy and CAM Dimension per Class", y=1.03, fontsize=18)
       
        plt.tight_layout()
        plt.show()
    
        plt.savefig(os.path.join(plot_path, f'r50_adv_acc_cam_dim{ext}c2.png'))

    return    


def multi_model_adv_manifold_dim(model1='ResNet50', model2='ConvNeXt', model3='VGG', n_classes=10):
    model1 = model_namer(model1)
    model2 = model_namer(model2)
    model3 = model_namer(model3)
    print(model1, model2, model3)

    ext = f'_{n_classes}' if n_classes != 10 else ''
    
    with open(save_path + f'{model1}_adv_accuracies{ext}c.pkl', 'rb') as f:
        m1_adv_accs = pickle.load(f)
    
    with open (save_path+ f'{model1}_class_manifold_dims{ext}_NN.pkl', 'rb') as f:
        m1_cam_dims = pickle.load(f)
    
    with open(save_path + f'{model2}_adv_accuracies{ext}c.pkl', 'rb') as f:
        m2_adv_accs = pickle.load(f)
    
    with open (save_path+f'{model2}_class_manifold_dims{ext}_NN.pkl', 'rb') as f:
        m2_cam_dims = pickle.load(f)
    
    with open(save_path + f'{model3}_adv_accuracies{ext}c.pkl', 'rb') as f:
        m3_adv_accs = pickle.load(f)
    
    with open (save_path+f'{model3}_class_manifold_dims{ext}_NN.pkl', 'rb') as f:
        m3_cam_dims = pickle.load(f)
    
    
    # sort the dictionaries by class name
    m1_adv_accs = dict(sorted(m1_adv_accs.items(), key=lambda item: item[0]))
    m1_cam_dims = dict(sorted(m1_cam_dims.items(), key=lambda item: item[0]))
    m2_adv_accs = dict(sorted(m2_adv_accs.items(), key=lambda item: item[0]))
    m2_cam_dims = dict(sorted(m2_cam_dims.items(), key=lambda item: item[0]))
    m3_adv_accs = dict(sorted(m3_adv_accs.items(), key=lambda item: item[0]))
    m3_cam_dims = dict(sorted(m3_cam_dims.items(), key=lambda item: item[0]))

    assert m1_adv_accs.keys() == m1_cam_dims.keys() == m2_adv_accs.keys() == m2_cam_dims.keys() == m3_adv_accs.keys() == m3_cam_dims.keys()

    class_labels = [index_to_label(int(cid)) for cid in m1_adv_accs.keys()]
    plot_df = pd.DataFrame({'class': class_labels, 'm1_adv_acc': list(m1_adv_accs.values()), 
                                'm1_cam_dim': list(m1_cam_dims.values()), 'm2_adv_acc': list(m2_adv_accs.values()),
                                'm2_cam_dim': list(m2_cam_dims.values()), 'm3_adv_acc': list(m3_adv_accs.values()),
                                'm3_cam_dim': list(m3_cam_dims.values())})
    

    epsilons = [0,0.0005,0.005]
    i = 1
    eps = epsilons[i]
    # print(plot_df['r50_adv_acc'])
    
    plot_df[f'm1_adv_acc'] = plot_df['m1_adv_acc'].apply(lambda x: x[i])
    plot_df['m1_adv_acc'] = plot_df[f'm1_adv_acc'].apply(lambda x: round(x, -1))
    plot_df[f'm2_adv_acc'] = plot_df['m2_adv_acc'].apply(lambda x: x[i])
    plot_df['m2_adv_acc'] = plot_df[f'm2_adv_acc'].apply(lambda x: round(x, -1))
    plot_df[f'm3_adv_acc'] = plot_df['m3_adv_acc'].apply(lambda x: x[i])
    plot_df['m3_adv_acc'] = plot_df[f'm3_adv_acc'].apply(lambda x: round(x, -1))

    # put all adv_accs in a column and add column for m1 and m2
    plot_df2 = pd.DataFrame(index=np.arange(len(plot_df)*3), columns=['adv_acc', 'cam_dim', 'model'])
    plot_df2['adv_acc'] = list(plot_df['m1_adv_acc']) + list(plot_df['m2_adv_acc']) + list(plot_df['m3_adv_acc'])
    plot_df2['cam_dim'] = list(plot_df['m1_cam_dim']) + list(plot_df['m2_cam_dim']) + list(plot_df['m3_cam_dim'])
    plot_df2['model'] = [model1 for _ in range(len(plot_df))] + [model2 for _ in range(len(plot_df))] + [model3 for _ in range(len(plot_df))]

    sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})
    rel = sns.relplot(data=plot_df2, x=f'adv_acc', y='cam_dim', kind='line', height=8, aspect=1.5, hue='model')

    # # add ale adv acc and cam dim
    # sns.relplot(data=plot_df, x=f'ale_adv_acc', y='ale_cam_dim', kind='line', height=8, aspect=1.5, color='red', ax=rel.ax)

    rel.set_axis_labels('Adversarial Accuracy (%)', 'CAM Dimension', fontsize=14)
    plt.title("Adversarial Accuracy and CAM Dimension per Class", fontsize=18, y=0.9, pad=20)

    plt.show()
    # # plt.suptitle(f'Pearson: {corr_coeff:.3f} p={float(p):.2e}', fontsize=14, y=0.90)
    plt.savefig(os.path.join(plot_path, f'many_models_adv_dim.png'))


def plot_manifold_dims(models=[]):
    # load the manifold dimensions for the models
    model_dims = []

    for model in models:
        mn = model_namer(model)
        with open(save_path + f'{mn}_class_manifold_dims_100.pkl', 'rb') as f:
            model_dims.append(pickle.load(f))
 
    # sort keys
    model_dims[0] = dict(sorted(model_dims[0].items(), key=lambda item: item[0]))
    model_dims[1] = dict(sorted(model_dims[1].items(), key=lambda item: item[0]))
    assert model_dims[0].keys() == model_dims[1].keys() 

    # access first key of model_dims


    plot_df = pd.DataFrame(index=np.arange(200), columns=['model', 'class', 'manifold_dim'])

    plot_df[f'model'] = [models[0] for _ in range(len(model_dims[0]))] + [models[1] for _ in range(len(model_dims[1]))]
    plot_df[f'class'] = [index_to_label(int(cid)) for cid in model_dims[0].keys()] + [index_to_label(int(cid)) for cid in model_dims[1].keys()]
    plot_df[f'manifold_dim'] = list(model_dims[0].values()) + list(model_dims[1].values())
    
    sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})
    rel = sns.relplot(data=plot_df, x='class', y='manifold_dim', kind='line', height=8, aspect=1.5, hue='model')
    rel.set_axis_labels('Classes', 'Manifold Dimension', fontsize=14)
    plt.title("Manifold Dimension per Class", fontsize=18, y=0.9, pad=20)
    # set y axis ticks
    plt.yticks(np.arange(35, 50, 1))
    plt.xticks([])
    plt.show()
    plt.savefig(os.path.join(plot_path, f'2-manifold_dims.png'))

    return

def plot_avg_dims(model1='ResNet50', model2='ConvNeXt', model3='VGG', model4='DenseNet', n_classes=10):
    model1 = model_namer(model1)
    model2 = model_namer(model2)
    model3 = model_namer(model3)
    model4 = model_namer(model4)
    print(model1, model2, model3, model4)

    ext = f'_{n_classes}' if n_classes != 10 else ''
    
    with open(save_path + f'{model1}_adv_accuracies{ext}c.pkl', 'rb') as f:
        m1_adv_accs = pickle.load(f)
    
    with open (save_path+ f'{model1}_class_manifold_dims{ext}.pkl', 'rb') as f:
        m1_cam_dims = pickle.load(f)
    
    with open(save_path + f'{model2}_adv_accuracies{ext}c.pkl', 'rb') as f:
        m2_adv_accs = pickle.load(f)
    
    with open (save_path+f'{model2}_class_manifold_dims{ext}.pkl', 'rb') as f:
        m2_cam_dims = pickle.load(f)
    
    with open(save_path + f'{model3}_adv_accuracies{ext}c.pkl', 'rb') as f:
        m3_adv_accs = pickle.load(f)
    
    with open (save_path+f'{model3}_class_manifold_dims{ext}.pkl', 'rb') as f:
        m3_cam_dims = pickle.load(f)
    
    with open(save_path + f'{model4}_adv_accuracies{ext}c.pkl', 'rb') as f:
        m4_adv_accs = pickle.load(f)
    
    with open (save_path+f'{model4}_class_manifold_dims{ext}.pkl', 'rb') as f:
        m4_cam_dims = pickle.load(f)
    

    # sort the dictionaries by class name
    m1_adv_accs = dict(sorted(m1_adv_accs.items(), key=lambda item: item[0]))
    m1_cam_dims = dict(sorted(m1_cam_dims.items(), key=lambda item: item[0]))
    m2_adv_accs = dict(sorted(m2_adv_accs.items(), key=lambda item: item[0]))
    m2_cam_dims = dict(sorted(m2_cam_dims.items(), key=lambda item: item[0]))
    m3_adv_accs = dict(sorted(m3_adv_accs.items(), key=lambda item: item[0]))
    m3_cam_dims = dict(sorted(m3_cam_dims.items(), key=lambda item: item[0]))
    m4_adv_accs = dict(sorted(m4_adv_accs.items(), key=lambda item: item[0]))
    m4_cam_dims = dict(sorted(m4_cam_dims.items(), key=lambda item: item[0]))


    assert m1_adv_accs.keys() == m1_cam_dims.keys() == m2_adv_accs.keys() == m2_cam_dims.keys() == m3_adv_accs.keys() == m3_cam_dims.keys() == m4_adv_accs.keys() == m4_cam_dims.keys()

    m1_avg_dim = np.mean(list(m1_cam_dims.values()))
    m2_avg_dim = np.mean(list(m2_cam_dims.values()))
    m3_avg_dim = np.mean(list(m3_cam_dims.values()))
    m4_avg_dim = np.mean(list(m4_cam_dims.values()))

    epsilons = [0,0.0005,0.005]
    i = 1
    eps = epsilons[i]

    m1_avg_acc = np.mean([x[i] for x in m1_adv_accs.values()])
    m2_avg_acc = np.mean([x[i] for x in m2_adv_accs.values()])
    m3_avg_acc = np.mean([x[i] for x in m3_adv_accs.values()])
    m4_avg_acc = np.mean([x[i] for x in m4_adv_accs.values()])


    plot_df = pd.DataFrame({'model': [model1, model2, model3, model4], 'avg_dim': [m1_avg_dim, m2_avg_dim, m3_avg_dim, m4_avg_dim], 'avg_acc': [m1_avg_acc, m2_avg_acc, m3_avg_acc, m4_avg_acc]})
    plot_df = plot_df.sort_values(by='avg_acc', ascending=False)

    # plt.figure(figsize=(15, 8))
    # ax = sns.barplot(data=plot_df, x='model', y='avg_acc')
    # ax2 = ax.twinx()
    # sns.lineplot(data=plot_df, x='model', y='avg_dim', ax=ax2, color='red', marker='o', lw=3)
   
    # plt.title("Adversarial Accuracy and CAM Dimension per Class", fontsize=18, pad=20)
    # plt.show()

    sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})
    pearson, p = pearsonr(plot_df['avg_acc'], plot_df['avg_dim'])

    rel = sns.relplot(data=plot_df, x='avg_acc', y='avg_dim', kind='scatter', hue='model', height=8, aspect=1.5)
    rel.set_axis_labels('Adversarial Accuracy (%)', 'CAM Dimension', fontsize=14)

    plt.title("Adversarial Accuracy and CAM Dimension per Class", fontsize=18, y=0.9, pad=20)
    plt.suptitle(f'Pearson: {pearson:.3f} p={float(p):.2e}', fontsize=15, y=0.88)
    plt.show()


    plt.savefig(os.path.join(plot_path, f'many_models_avg_dim_avg_acc.png'))

    
    return

def plot_explained_var():
    with open(save_path + 'r50_class_activations_100c.pkl', 'rb') as f:
        class_acts = pickle.load(f)

    # sort class_acts
    class_acts = dict(sorted(class_acts.items(), key=lambda item: item[0]))

    activations = np.array(list(class_acts.values()))
    class_indices = np.array(list(class_acts.keys()))

    index = 0
    class_label= index_to_label(int(class_indices[index]))
    class_acts = activations[index]


    # plot 4 classes on the same plot
    classes = [0, 1, 2, 3]
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    for i, ax in enumerate(axs.flat):
        class_label = index_to_label(int(class_indices[classes[i]]))
        class_acts = activations[classes[i]]
        pca = PCA()
        # preprocess the data
        scaler = StandardScaler()
        class_acts = scaler.fit_transform(class_acts)
        pca.fit(class_acts)
        var = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(var, marker='o')
        ax.axhline(0.95, color='black', linestyle='--', alpha=0.5)
        ax.axvline(np.argmax(var > 0.95), color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{class_label.capitalize()}', fontsize=14)
        ax.set_xlabel('Number of Components', fontsize=13)
        ax.set_ylabel('Explained Variance', fontsize=13)
        ax.set_xticks(np.arange(0, 60, 5))
        ax.set_yticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_path, f'4_classes_explained_var.png'))
    return    

def plot_avg_dims_diff_thresholds():
    with open(save_path + 'r50_class_activations_100c.pkl', 'rb') as f:
        class_acts = pickle.load(f)
    
    # sort class_acts
    class_acts = dict(sorted(class_acts.items(), key=lambda item: item[0]))

    activations = np.array(list(class_acts.values()))
    class_inds = np.array(list(class_acts.keys()))
    
    thresholds = [0.8, 0.85, 0.9, 0.95, 0.999]

    thresh_dict = {t: None for t in thresholds}

    # barplots for each threshold where y is the avg dim
    for thresh in thresholds:
        class_dims = []
        for i in range(100):
            class_acts = activations[i]
            pca = PCA()
            # preprocess the data
            scaler = StandardScaler()
            class_acts = scaler.fit_transform(class_acts)
            pca.fit(class_acts)
            var = np.cumsum(pca.explained_variance_ratio_)
            class_dims.append(np.argmax(var >= thresh))
        thresh_dict[thresh] = np.mean(class_dims)
        
    thresh_dict['layer'] = 60
    
    sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})

    plot_df = pd.DataFrame({'threshold': list(thresh_dict.keys()), 'avg_dim': list(thresh_dict.values())})

    sns.barplot(data=plot_df, x='threshold', y='avg_dim')
    # change colour of layer bar
    plt.bar(4, 60, color='red')
    # label the layer bar
    plt.text(4, 60, '2048', ha='center', va='bottom', fontsize=12)

    plt.ylabel('Average CAM Dimension')
    plt.xlabel('Threshold for PCA')
    plt.yticks(np.arange(0, 55, 5))

    plt.title("Average CAM Dimension for Different Thresholds", fontsize=18, pad=20)
    plt.show()
    plt.savefig(os.path.join(plot_path, f'avg_dim_diff_thresholds.png'))


def plot_models_adv():
    models = ['ResNet50', 'ConvNeXt', 'VGG16', 'AlexNet']
    # barplot for each model where y is the avg adv acc
    adv_accs_dict = {model: None for model in models}
    cam_dims_dict = {model: None for model in models}
    
    for model in models:
        mn = model_namer(model)
        with open(save_path + f'{mn}_adv_accuracies_100c.pkl', 'rb') as f:
            adv_accs = pickle.load(f)
        eps_accs = [x[1] for x in adv_accs.values()]
        avg_acc = np.mean(eps_accs)
        adv_accs_dict[model] = avg_acc

        with open(save_path + f'{mn}_class_manifold_dims_100.pkl', 'rb') as f:
            cam_dims = pickle.load(f)
        avg_dim = np.mean(list(cam_dims.values()))
        cam_dims_dict[model] = avg_dim
    # sort both dicts by model name
    adv_accs_dict = dict(sorted(adv_accs_dict.items(), key=lambda item: item[0]))
    cam_dims_dict = dict(sorted(cam_dims_dict.items(), key=lambda item: item[0]))
    print(adv_accs_dict)
    print(cam_dims_dict)
    plot_df = pd.DataFrame({'model': list(adv_accs_dict.keys()), 'avg_acc': list(adv_accs_dict.values()), 'avg_dim': list(cam_dims_dict.values())})
    # sort plot df by avg acc
    plot_df = plot_df.sort_values(by='avg_dim', ascending=False)
    
    # plot a barplot to show the avg adv acc for each model and a twin lineplot for the avg cam dim on the same plot
    sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})
    rel = sns.barplot(data=plot_df, x='model', y='avg_acc')
    # set axis labels
    rel.set_ylabel('Average Adversarial Accuracy (%)', fontsize=14)
    rel.set_xlabel('Models', fontsize=14)
    rel.ax2 = rel.twinx()
    sns.lineplot(data=plot_df, x='model', y='avg_dim', ax=rel.ax2, color='red', marker='o', lw=3)
    rel.ax2.set_ylabel('Average CAM Dimension', fontsize=14)
    plt.title("Average Adversarial Accuracy and CAM Dimension per Class", fontsize=18, pad=20)
    plt.show()
    plt.savefig(os.path.join(plot_path, f'models_avg_adv_dim.png'))

    # clear plt
    plt.clf()


    # other way around
    # plot a barplot to show the avg adv acc for each model and a twin lineplot for the avg cam dim on the same plot
    sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})
    plt.grid(False)
    rel = sns.barplot(data=plot_df, x='model', y='avg_dim')
    # set axis labels
    rel.set_ylabel('Average CAM Dimension', fontsize=14)
    rel.set_xlabel('Models', fontsize=14)
    rel.set_ylim(30,45)
    rel.ax2 = rel.twinx()
    sns.lineplot(data=plot_df, x='model', y='avg_acc', ax=rel.ax2, color='red', marker='o', lw=3, label="Avg Adv. Accuracy")
    rel.ax2.set_ylabel('Average Adversarial Accuracy (%)', fontsize=14)
    plt.title("Average Adversarial Accuracy and CAM Dimension per Class", fontsize=18, pad=20)
    # turn off gridlines
    plt.grid(False)
    # legend on
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(plot_path, f'models_avg_adv_dim2.png'))


    return

if __name__ == '__main__':
    # plot_class_accs()
    # plot_class_accdiff_manifold_dim()
    # plot_class_acc_manifold_dim('train')
    # plot_adv_accs(model='VGG', n_classes=100) 
    # plot_adv_acc_manifold_dim(100, all_eps=False)
    # multi_model_adv_manifold_dim(model1='ResNet50', model2='ConvNeXt', model3='DenseNet', n_classes=100)
    # plot_manifold_dims(models=['ResNet50', 'RobustResNet50'])
    # plot_avg_dims(model1='ResNet50', model2='ConvNeXt', model3='DenseNet', model4='VGG16', n_classes=100)

    # plot_explained_var()
    # plot_avg_dims_diff_thresholds()
    plot_models_adv()
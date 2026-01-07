import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

def plot_pca(df, title, save_dir):
    X = df.drop(columns=['disease_status'])
    y = df['disease_status']
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    plot_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df.index)
    plot_df['disease_status'] = y.values
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid', font_scale=1.2)
    palette = {0: '#0E87A6', 1: '#B7094C'}
    sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='disease_status',
        palette=palette,
        hue_order=[0, 1],
        s=120,
        alpha=0.9
    )
    plt.title(title + ' PCA', fontsize=18, weight='bold')
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.legend(title='Disease Status', loc='best')
    sns.despine(trim=True)
    save_path = f'{save_dir}/{title.replace(" ", "_").lower()}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne(df, title, save_dir):
    X = df.drop(columns=['disease_status'])
    y = df['disease_status']
    X_scaled = StandardScaler().fit_transform(X)
    X_tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='pca').fit_transform(X_scaled)
    plot_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'], index=df.index)
    plot_df['disease_status'] = y.values
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid', font_scale=1.2)
    palette = {0: '#0E87A6', 1: '#B7094C'}
    sns.scatterplot(
        data=plot_df,
        x='TSNE1',
        y='TSNE2',
        hue='disease_status',
        palette=palette,
        hue_order=[0, 1],
        s=120,
        alpha=0.9
    )
    plt.title(title + ' t-SNE', fontsize=18, weight='bold')
    plt.xlabel('TSNE1', fontsize=14)
    plt.ylabel('TSNE2', fontsize=14)
    plt.legend(title='Disease Status', loc='best', frameon=True, shadow=True)
    sns.despine(trim=True)
    save_path = f'{save_dir}/{title.replace(" ", "_").lower()}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_umap(df, title, save_dir):
    y = df['disease_status']
    X = df.drop(columns=['disease_status'])
    X_scaled = StandardScaler().fit_transform(X)
    X_umap = UMAP(n_components=2, random_state=42).fit_transform(X_scaled)
    plot_df = df.copy()
    plot_df['UMAP1'] = X_umap[:, 0]
    plot_df['UMAP2'] = X_umap[:, 1]
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.set(style='whitegrid', font_scale=1.2)
    palette = {0: '#0E87A6', 1: '#B7094C'}
    sns.scatterplot(
        data=plot_df,
        x='UMAP1',
        y='UMAP2',
        hue='disease_status',
        palette=palette,
        hue_order=[0, 1],
        s=120,
        alpha=0.9
    )
    plt.title(title + ' UMAP', fontsize=18, weight='bold')
    plt.xlabel('UMAP1', fontsize=14)
    plt.ylabel('UMAP2', fontsize=14)
    plt.legend(title='Disease Status', loc='best', frameon=True, shadow=True)
    sns.despine(trim=True)
    save_path = f'{save_dir}/{title.replace(" ", "_").lower()}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

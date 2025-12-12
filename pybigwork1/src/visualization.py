# 将以下代码复制到 src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置通用绘图风格
plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False 

def plot_price_distribution(df, save_dir=None):
    """画房价分布图"""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True, bins=50)
    plt.title('房价分布 (Price)')
    plt.xlabel('价格')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'price_distribution.png'))
        print(f"图片已保存: {save_dir}/price_distribution.png")
    plt.show()

def plot_correlation_heatmap(df, save_dir=None):
    """画热力图"""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('特征相关性热力图')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'))
        print(f"图片已保存: {save_dir}/correlation_heatmap.png")
    plt.show()

def plot_true_vs_pred(y_true, y_pred, save_dir=None):
    """画预测结果对比图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('真实房价')
    plt.ylabel('预测房价')
    plt.title('预测结果对比')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'prediction_vs_true.png'))
        print(f"图片已保存: {save_dir}/prediction_vs_true.png")
    plt.show()
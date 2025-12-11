import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_attention_logic():
    # 模拟一个 9x9 的注意力权重矩阵
    # 0号是 Global，1-8号是 Experts
    N = 9
    attention_matrix = np.zeros((N, N))

    # 设置模拟数值
    # 1. 对角线（自我关注）：通常很高
    np.fill_diagonal(attention_matrix, 0.8)

    # 2. Row 0 (Global -> Experts): 全局上下文关注某些特定专家（比如专家3和专家7）
    attention_matrix[0, 3] = 0.6  # 关注专家3
    attention_matrix[0, 7] = 0.5  # 关注专家7

    # 3. Col 0 (Experts -> Global): 所有专家都在一定程度上参考全局信息
    attention_matrix[1:, 0] = 0.4  # 大家都看指挥官

    # 4. Expert -> Expert: 专家1和专家2协作
    attention_matrix[1, 2] = 0.3
    attention_matrix[2, 1] = 0.3

    # 绘图
    plt.figure(figsize=(10, 8))
    labels = ['Global'] + [f'Exp {i}' for i in range(1, 9)]

    sns.heatmap(attention_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels)

    plt.title("Visualized Logic of Interaction Attention (9x9 Matrix)", fontsize=15)
    plt.xlabel("Key (Source of Info)", fontsize=12)
    plt.ylabel("Query (Seeker of Info)", fontsize=12)

    # 框出重点区域
    # Global -> Experts (Row 0)
    plt.gca().add_patch(plt.Rectangle((1, 0), 8, 1, fill=False, edgecolor='red', lw=3, label='Global Queries Experts'))
    # Experts -> Global (Col 0)
    plt.gca().add_patch(plt.Rectangle((0, 1), 1, 8, fill=False, edgecolor='blue', lw=3, label='Experts Query Global'))

    plt.show()


# 运行此函数即可生成图表
if __name__ == "__main__":
    plot_attention_logic()
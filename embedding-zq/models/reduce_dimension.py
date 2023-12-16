from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np


def reduce_dimension(input, output_n, reduce_type='pca', **kargs):
    print(f'开始降维, 降维方法 {reduce_type}, 输入维度 {input.shape[0]}, 输出维度 {output_n}')
    assert isinstance(input, np.ndarray)
    if input.shape[0] < input.shape[1]:
        print(f'WARNING, 输入向量维度为{input.shape}, 输入维度应为(samples, n_features)')

    if reduce_type == 'tsne':
        output = TSNE(n_components=output_n, init='pca').fit_transform(input)
    elif reduce_type == 'pca':
        output = PCA(n_components=output_n).fit_transform(input)
    elif reduce_type == 'pca_tsne':
        pca = PCA(n_components=0.99)
        x = pca.fit_transform(input)
        print(f'pca noise_variance {pca.noise_variance_}')
        tsne = TSNE(n_components=output_n, init='pca')
        output = tsne.fit_transform(x)
    else:
        raise Exception('ERROR, reduce_type must in tsne/pca/pca_tsne')
    print(f'降维完成')
    return output


if __name__ == "__main__":

    reduce_dimension(np.array([[1,2],[2,3],[3,4]]), 2)

    # 加载数据集
    # iris = load_iris()
    # tsne = use_tsne_reduce_dimension(iris.data)
    # pca = use_pca_reduce_dimension(iris.data)
    #
    # ddd = pca_tsne(iris.data)
    #
    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=iris.target)
    # plt.subplot(122)
    # plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
    # plt.colorbar()
    # plt.savefig('b.png')


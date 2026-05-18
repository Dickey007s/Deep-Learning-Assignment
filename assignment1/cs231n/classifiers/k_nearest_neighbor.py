from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ 使用 L2 距离（欧氏距离）的 kNN 分类器 """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练分类器。对于 k-近邻算法，这只是简单地记忆训练数据。

        输入：
        - X: 形状为 (num_train, D) 的 numpy 数组，包含训练数据
             num_train 个样本，每个样本有 D 个特征
        - y: 形状为 (N,) 的 numpy 数组，包含训练标签
             y[i] 是 X[i] 对应的标签
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        使用此分类器预测测试数据的标签。

        输入：
        - X: 形状为 (num_test, D) 的 numpy 数组，包含测试数据
             num_test 个样本，每个样本有 D 个特征
        - k: 参与投票的最近邻居数量
        - num_loops: 决定使用哪种方式计算距离（0/1/2层循环）

        返回：
        - y: 形状为 (num_test,) 的 numpy 数组，包含预测的标签
             y[i] 是测试点 X[i] 的预测标签
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("无效的 num_loops 值: %d" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        使用双层循环计算每个测试点和每个训练点之间的距离。

        输入：
        - X: 形状为 (num_test, D) 的测试数据

        返回：
        - dists: 形状为 (num_test, num_train) 的距离矩阵
                 dists[i, j] 是第 i 个测试点和第 j 个训练点之间的欧氏距离
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # 计算第 i 个测试点和第 j 个训练点之间的 L2 距离，                        #
                # 并将结果存入 dists[i, j]。                                          #
                # 不要使用维度循环，也不要使用 np.linalg.norm()。                        #
                #####################################################################
                dists[i,j] = np.sqrt(np.sum((X[i]-self.X_train[j])**2))
                pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        使用单层循环计算距离（只遍历测试数据）。

        输入/输出：与 compute_distances_two_loops 相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # 计算第 i 个测试点与所有训练点之间的 L2 距离，                       #
            # 并将结果存入 dists[i, :]。                                         #
            # 不要使用 np.linalg.norm()。                                        #
            #######################################################################
            diff = X[i] - self.X_train
            sum_sq = np.sum(diff**2, axis=1)
            dists[i,:] = np.sqrt(sum_sq)

            pass
        return dists

    def compute_distances_no_loops(self, X):
        """
        不使用显式循环计算所有测试点和训练点之间的距离。

        输入/输出：与 compute_distances_two_loops 相同

        提示：尝试用公式 √(a² + b² - 2ab) 将 L2 距离转化为矩阵运算
              使用矩阵乘法和两次广播求和。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # 不使用任何显式循环，计算所有测试点和所有训练点之间的 L2 距离，         #
        # 并将结果存入 dists。                                                  #
        #                                                                       #
        # 只使用基本的数组操作实现此函数；                                      #
        # 特别注意不要使用 scipy 的函数，                                       #
        # 也不要使用 np.linalg.norm()。                                         #
        #                                                                       #
        # 提示：尝试将 L2 距离公式转化为矩阵乘法                                 #
        #       和两次广播求和。                                                #
        #########################################################################
        X_train_sq = np.sum(self.X_train**2, axis=1) # 
        X_test_sq = np.sum(X**2, axis=1)
        AmB = X @ self.X_train.T

        # dists = np.sqrt(X_train_sq + X_test_sq - 2*AmB) 但是需要先进行维度对齐
        dists = np.sqrt(X_train_sq[np.newaxis,:] + X_test_sq[:,np.newaxis] - 2*AmB)

        return dists

    def predict_labels(self, dists, k=1):
        """
        根据距离矩阵预测每个测试点的标签。

        输入：
        - dists: 形状为 (num_test, num_train) 的距离矩阵
                 dists[i, j] 是第 i 个测试点和第 j 个训练点之间的距离

        返回：
        - y: 形状为 (num_test,) 的 numpy 数组，包含预测的标签
             y[i] 是测试点 X[i] 的预测标签
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 存储 k 个最近邻居的标签
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # 使用距离矩阵找到第 i 个测试点的 k 个最近邻居，                        #
            # 使用 self.y_train 找到这些邻居的标签。                                #
            # 将这些标签存入 closest_y。                                            #
            # 提示：查阅 numpy.argsort 函数。                                       #
            #########################################################################
            closest_idx = np.argsort(dists[i])[:k]

            #########################################################################
            # TODO:                                                                 #
            # 找到 closest_y 中最常见的标签，                                        #
            # 将此标签存入 y_pred[i]。                                              #
            # 如果有平票，选择较小的标签。                                          #
            #########################################################################
            closest_y = self.y_train[closest_idx]
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred
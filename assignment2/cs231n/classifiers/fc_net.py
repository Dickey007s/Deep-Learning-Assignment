from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """多层全连接神经网络类。

    网络包含任意数量的隐藏层、ReLU 非线性激活函数和 softmax 损失函数。
    还可选实现 dropout 和 batch/layer normalization。
    对于包含 L 层的网络，其架构为：

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    其中 batch/layer normalization 和 dropout 是可选的，{...} 块重复 L - 1 次。

    可学习参数存储在 self.params 字典中，并将通过 Solver 类进行学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """初始化一个新的 FullyConnectedNet。

        输入参数：
        - hidden_dims: 整数列表，指定每个隐藏层的大小。
        - input_dim: 整数，指定输入的大小。
        - num_classes: 整数，指定要分类的类别数量。
        - dropout_keep_ratio: 0 到 1 之间的标量，指定 dropout 强度。
            如果 dropout_keep_ratio=1，则网络完全不使用 dropout。
        - normalization: 网络使用的归一化类型。有效值为
            "batchnorm"、"layernorm" 或 None（默认值，不使用归一化）。
        - reg: 标量，指定 L2 正则化强度。
        - weight_scale: 标量，指定权重随机初始化的标准差。
        - dtype: numpy 数据类型对象；所有计算将使用此数据类型执行。
            float32 更快但精度较低，因此数值梯度检查时应使用 float64。
        - seed: 如果不为 None，则将此随机种子传递给 dropout 层。
            这将使 dropout 层具有确定性，以便我们可以对模型进行梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: 初始化网络的参数，将所有值存储在 self.params 字典中。第一层          #
        # 的权重和偏置存储在 W1 和 b1 中；第二层使用 W2 和 b2，以此类推。权重应         #
        # 该从均值为 0、标准差等于 weight_scale 的正态分布中初始化。偏置应初始化为      #
        # 零。                                                                      #
        #                                                                          #
        # 使用 batch normalization 时，第一层的缩放和平移参数存储在 gamma1 和          #
        # beta1 中；第二层使用 gamma2 和 beta2，以此类推。缩放参数应初始化为 1，        #
        # 平移参数应初始化为 0。                                                     #
        ############################################################################
        last_layer = len(hidden_dims) + 1

        self.params[f'W{1}'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        self.params[f'b{1}'] = np.zeros(hidden_dims[0])

        for i in range(2, last_layer):
            self.params[f'W{i}'] = weight_scale * np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
            self.params[f'b{i}'] = np.zeros(hidden_dims[i-1])
            
        self.params[f'W{last_layer}'] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        self.params[f'b{last_layer}'] = np.zeros(num_classes)

        if self.normalization is not None:
            for i in range(1, last_layer):
                self.params[f'gamma{i}'] = np.ones(hidden_dims[i-1])
                self.params[f'beta{i}'] = np.zeros(hidden_dims[i-1])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 使用 dropout 时，需要向每个 dropout 层传递一个 dropout_param 字典，
        # 以便该层知道 dropout 概率和模式（train / test）。
        # 你可以将相同的 dropout_param 传递给每个 dropout 层。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # 使用 batch normalization 时，需要跟踪 running means 和 variances，
        # 因此需要向每个 batch normalization 层传递一个特殊的 bn_param 对象。
        # 应将 self.bn_params[0] 传递给第一个 batch normalization 层的前向传播，
        # self.bn_params[1] 传递给第二个 batch normalization 层的前向传播，以此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # 将所有参数转换为正确的数据类型。
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """计算全连接网络的损失和梯度。

        输入参数：
        - X: 输入数据数组，形状为 (N, d_1, ..., d_k)
        - y: 标签数组，形状为 (N,)。y[i] 给出 X[i] 的标签。

        返回值：
        如果 y 为 None，则运行模型测试时的前向传播并返回：
        - scores: 形状为 (N, C) 的数组，给出分类分数，其中
            scores[i, c] 是 X[i] 和类别 c 的分类分数。

        如果 y 不为 None，则运行训练时的前向和反向传播并返回一个元组：
        - loss: 给出损失的标量值
        - grads: 与 self.params 键相同的字典，将参数名映射到损失
            相对于这些参数的梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # 为 batchnorm 参数和 dropout 参数设置 train/test 模式，
        # 因为它们在训练和测试期间的行为不同。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: 实现全连接网络的前向传播，计算 X 的类别分数并存储在 scores 变量中。    #
        #                                                                          #
        # 使用 dropout 时，需要将 self.dropout_param 传递给每个 dropout 前向传播。    #
        #                                                                          #
        # 使用 batch normalization 时，需要将 self.bn_params[0] 传递给第一个          #
        # batch normalization 层的前向传播，self.bn_params[1] 传递给第二个            #
        # batch normalization 层的前向传播，以此类推。                               #
        ############################################################################
        cache = {}


        h = X
        for i in range(1, self.num_layers):
            a, fc_cache = affine_forward(h, self.params[f'W{i}'], self.params[f'b{i}'])
            if self.normalization is not None:
                bn, bn_cache = batchnorm_forward(a, self.params[f'gamma{i}'], self.params[f'beta{i}'], self.bn_params[i-1])
            else:
                bn = a
                bn_cache = None
            h, relu_cache = relu_forward(bn)
            cache[i] = (fc_cache, bn_cache, relu_cache)

        scores, cache[self.num_layers] = affine_forward(
            h,
            self.params[f'W{self.num_layers}'],
            self.params[f'b{self.num_layers}']
        )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # 如果是测试模式，提前返回。
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: 实现全连接网络的反向传播。将损失存储在 loss 变量中，梯度存储在         #
        # grads 字典中。使用 softmax 计算数据损失，并确保 grads[k] 保存              #
        # self.params[k] 的梯度。不要忘记添加 L2 正则化！                            #
        #                                                                          #
        # 使用 batch/layer normalization 时，不需要对缩放和平移参数进行正则化。        #
        #                                                                          #
        # 注意：为了确保你的实现与我们的匹配并通过自动测试，请确保你的 L2 正则化        #
        # 包含一个 0.5 的因子，以简化梯度的表达式。                                  #
        ############################################################################
        loss, ds = softmax_loss(scores, y)
        dh, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = affine_backward(ds, cache[self.num_layers])

        for i in range(self.num_layers - 1, 0, -1):
            fc_cache, bn_cache, relu_cache = cache[i]
            da = relu_backward(dh, relu_cache)
            if self.normalization is not None:
                da, grads[f'gamma{i}'], grads[f'beta{i}'] = batchnorm_backward(da, bn_cache)
            dh, grads[f'W{i}'], grads[f'b{i}'] = affine_backward(da, fc_cache)

        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f'W{i}'] ** 2)
            grads[f'W{i}'] += self.reg * self.params[f'W{i}']

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

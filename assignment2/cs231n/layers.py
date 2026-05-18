from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """计算仿射（全连接）层的前向传播。

    输入 x 的形状为 (N, d_1, ..., d_k)，包含 N 个样本的 minibatch，
    其中每个样本 x[i] 的形状为 (d_1, ..., d_k)。我们将每个输入重塑为
    维度 D = d_1 * ... * d_k 的向量，然后将其变换为维度 M 的输出向量。

    输入:
    - x: 包含输入数据的 numpy 数组，形状为 (N, d_1, ..., d_k)
    - w: 权重 numpy 数组，形状为 (D, M)
    - b: 偏置 numpy 数组，形状为 (M,)

    返回一个元组:
    - out: 输出，形状为 (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    out = x_reshaped.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """计算仿射（全连接）层的反向传播。

    输入:
    - dout: 上游梯度，形状为 (N, M)
    - cache: 元组，包含:
      - x: 输入数据，形状为 (N, d_1, ... d_k)
      - w: 权重，形状为 (D, M)
      - b: 偏置，形状为 (M,)

    返回一个元组:
    - dx: 关于 x 的梯度，形状为 (N, d1, ..., d_k)
    - dw: 关于 w 的梯度，形状为 (D, M)
    - db: 关于 b 的梯度，形状为 (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)

    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_reshaped.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """计算修正线性单元（ReLU）层的前向传播。

    输入:
    - x: 输入，任意形状

    返回一个元组:
    - out: 输出，与 x 形状相同
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """计算修正线性单元（ReLU）层的反向传播。

    输入:
    - dout: 上游梯度，任意形状
    - cache: 输入 x，与 dout 形状相同

    返回:
    - dx: 关于 x 的梯度
    """
    dx, x = None, cache
    dx = dout * (x > 0)
    return dx


def softmax_loss(x, y):
    """计算 softmax 分类的损失和梯度。

    输入:
    - x: 输入数据，形状为 (N, C)，其中 x[i, j] 是第 i 个输入在第 j 个类别上的得分。
    - y: 标签向量，形状为 (N,)，其中 y[i] 是 x[i] 的标签，且 0 <= y[i] < C

    返回一个元组:
    - loss: 标量，表示损失
    - dx: 关于 x 的损失梯度
    """
    N = x.shape[0]
    x_shift = x - np.max(x, axis=1, keepdims=True)
    probs = np.exp(x_shift) / np.sum(np.exp(x_shift), axis=1, keepdims=True)
    loss = np.mean(-np.log(probs[np.arange(N), y]))
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """批归一化的前向传播。

    在训练期间，样本均值和（未修正的）样本方差从 minibatch 统计量中计算，
    并用于归一化输入数据。在训练期间，我们还对每个特征的均值和方差保持
    指数衰减的运行均值，这些平均值用于测试时归一化数据。

    在每个时间步，我们使用基于动量参数的指数衰减来更新均值和方差的运行平均值:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    注意，批归一化论文建议了不同的测试时行为：他们使用大量训练图像
    计算每个特征的样本均值和方差，而不是使用运行平均值。对于这个实现，
    我们选择使用运行平均值，因为它们不需要额外的估计步骤；torch7 中
    批归一化的实现也使用运行平均值。

    输入:
    - x: 数据，形状为 (N, D)
    - gamma: 缩放参数，形状为 (D,)
    - beta: 平移参数，形状为 (D,)
    - bn_param: 字典，包含以下键:
      - mode: 'train' 或 'test'；必需
      - eps: 数值稳定性常数
      - momentum: 运行均值/方差的常数
      - running_mean: 形状为 (D,) 的数组，给出特征的运行均值
      - running_var: 形状为 (D,) 的数组，给出特征的运行方差

    返回一个元组:
    - out: 形状为 (N, D)
    - cache: 反向传播所需的值组成的元组
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: 实现批归一化的训练时前向传播。                                  
        # 使用 minibatch 统计量计算均值和方差，使用这些统计量归一化输入数据，   
        # 并使用 gamma 和 beta 对归一化后的数据进行缩放和平移。                 
        #                                                                     
        # 你应该将输出存储在变量 out 中。反向传播所需的任何中间变量都应存储在   
        # cache 变量中。                                                       
        #                                                                    
        # 你还应该使用计算出的样本均值和方差以及动量变量来更新运行均值和运行   
        # 方差，将结果存储在 running_mean 和 running_var 变量中。               
        #                                                                    
        # 注意，虽然你应该跟踪运行方差，但应该基于标准差（方差的平方根）来     
        # 归一化数据！                                                         
        # 参考原始论文 (https://arxiv.org/abs/1502.03167) 可能会有帮助。        
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var  = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps) 
        out = x_hat * gamma + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var  = momentum * running_var  + (1 - momentum) * sample_var

        cache = (x, x_hat, sample_mean, sample_var, gamma, beta, eps)

        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: 实现批归一化的测试时前向传播。                                 
        # 使用运行均值和方差归一化输入数据，然后使用 gamma 和 beta 对归一化后   
        # 的数据进行缩放和平移。将结果存储在 out 变量中。                       
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var  = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps) 
        out = x_hat * gamma + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var  = momentum * running_var  + (1 - momentum) * sample_var

        cache = (x, x_hat, sample_mean, sample_var, gamma, beta, eps)

        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # 将更新后的运行均值存回 bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """批归一化的反向传播。

    对于这个实现，你应该在纸上写出批归一化的计算图，
    并通过中间节点反向传播梯度。

    输入:
    - dout: 上游梯度，形状为 (N, D)
    - cache: 来自 batchnorm_forward 的中间变量。

    返回一个元组:
    - dx: 关于输入 x 的梯度，形状为 (N, D)
    - dgamma: 关于缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta: 关于平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 实现批归一化的反向传播。将结果存储在 dx、dgamma 和 dbeta 变量中。 #
    # 参考原始论文 (https://arxiv.org/abs/1502.03167) 可能会有帮助。          #
    ###########################################################################
    x, x_hat, sample_mean, sample_var, gamma, beta, eps = cache

    N = x.shape[0]

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_hat = dout * gamma

    dx = dx_hat / np.sqrt(sample_var + eps)
    dvar = np.sum(dx_hat * (x - sample_mean) * (-0.5) * (sample_var + eps)**(-3/2) , axis=0)
    dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis=0) - dvar * (-2) * np.mean(x - sample_mean)

    dx += dmean / N + dvar * 2.0/N * (x - sample_mean)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """批归一化的替代反向传播。

    对于这个实现，你应该在纸上推导出批归一化反向传播的导数，
    并尽可能简化。你应该能够推导出一个简单的反向传播表达式。
    参见 jupyter notebook 获取更多提示。

    注意：这个实现应该接收与 batchnorm_backward 相同的 cache 变量，
    但可能不会使用 cache 中的所有值。

    输入/输出: 与 batchnorm_backward 相同
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 实现批归一化的反向传播。将结果存储在 dx、dgamma 和 dbeta 变量中。 #
    #                                                                         #
    # 在计算关于中心化输入的梯度后，你应该能够在一个语句中计算出关于输入的    #
    # 梯度；我们的实现可以放在单行 80 字符内。                                #
    ###########################################################################
    x, x_hat, sample_mean, sample_var, gamma, beta, eps = cache

    N = x.shape[0]

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_hat = dout * gamma

    dx = dx_hat / np.sqrt(sample_var + eps)
    dvar = np.sum(dx_hat * (x - sample_mean) * (-0.5) * (sample_var + eps)**(-3/2) , axis=0)
    dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis=0) - dvar * (-2) * np.mean(x - sample_mean)

    dx += dmean / N + dvar * 2.0/N * (x - sample_mean)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """层归一化的前向传播。

    在训练和测试期间，输入数据在每个数据点上进行归一化，
    然后使用与批归一化相同的 gamma 和 beta 参数进行缩放和平移。

    注意，与批归一化不同，层归一化在训练和测试期间的行为是相同的，
    我们不需要跟踪任何运行平均值。

    输入:
    - x: 数据，形状为 (N, D)
    - gamma: 缩放参数，形状为 (D,)
    - beta: 平移参数，形状为 (D,)
    - ln_param: 字典，包含以下键:
        - eps: 数值稳定性常数

    返回一个元组:
    - out: 形状为 (N, D)
    - cache: 反向传播所需的值组成的元组
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: 实现层归一化的训练时前向传播。                                    #
    # 归一化输入数据，并使用 gamma 和 beta 对归一化后的数据进行缩放和平移。     #
    # 提示：这可以通过稍微修改你的训练时批归一化实现，并插入一两行位置恰当    #
    # 的代码来完成。特别是，你能想到任何可以执行的矩阵变换，使你能够复制     #
    # 批归一化代码并几乎不做修改吗？                                          #
    ###########################################################################
    mean = np.mean(x, axis=1)
    var  = np.var(x, axis=1)

    x_hat = x - mean / np.sqrt(var + eps)
    out =  x_hat * gamma + beta

    cache = (x, x_hat, mean, var, gamma, beta, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """层归一化的反向传播。

    对于这个实现，你可以大量依赖你已经完成的批归一化工作。

    输入:
    - dout: 上游梯度，形状为 (N, D)
    - cache: 来自 layernorm_forward 的中间变量。

    返回一个元组:
    - dx: 关于输入 x 的梯度，形状为 (N, D)
    - dgamma: 关于缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta: 关于平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: 实现层归一化的反向传播。                                          #
    #                                                                         #
    # 提示：这可以通过稍微修改你的训练时批归一化实现来完成。前向传播的提示    #
    # 仍然适用！                                                              #
    ###########################################################################
    x, x_hat, mean, var, gamma, beta, eps = cache

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta  = np.sum(dout, axis=0)
    dx_hat = dout * gamma

    std = np.sqrt(var + eps)
    dx = dx_hat / std
    dvar = dx_hat * (-0.5) * (x - mean) / (std)**(3/2)
    dmean = (dvar/dmean) + (dx_hat/dmean)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """反向 dropout 的前向传播。

    注意，这与普通版本的 dropout 不同。
    这里，p 是保留神经元输出的概率，而不是丢弃神经元输出的概率。
    详见 http://cs231n.github.io/neural-networks-2/#reg。

    输入:
    - x: 输入数据，任意形状
    - dropout_param: 字典，包含以下键:
      - p: Dropout 参数。我们以概率 p 保留每个神经元输出。
      - mode: 'test' 或 'train'。如果模式是 train，则执行 dropout；
        如果模式是 test，则直接返回输入。
      - seed: 随机数生成器的种子。传递种子使此函数具有确定性，
        这对于梯度检查是必需的，但在真实网络中不需要。

    输出:
    - out: 与 x 形状相同的数组。
    - cache: 元组 (dropout_param, mask)。在训练模式下，mask 是用于乘以
      输入的 dropout mask；在测试模式下，mask 为 None。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: 实现反向 dropout 的训练阶段前向传播。                             #
        # 将 dropout mask 存储在 mask 变量中。                                  #
        #######################################################################
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: 实现反向 dropout 的测试阶段前向传播。                             #
        #######################################################################
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """反向 dropout 的反向传播。

    输入:
    - dout: 上游梯度，任意形状
    - cache: 来自 dropout_forward 的 (dropout_param, mask)。
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: 实现反向 dropout 的训练阶段反向传播                               #
        #######################################################################
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """卷积层前向传播的朴素实现。

    输入包含 N 个数据点，每个数据点有 C 个通道，高度 H 和宽度 W。
    我们用 F 个不同的滤波器对每个输入进行卷积，每个滤波器跨越所有 C 个通道，
    高度为 HH，宽度为 WW。

    输入:
    - x: 输入数据，形状为 (N, C, H, W)
    - w: 滤波器权重，形状为 (F, C, HH, WW)
    - b: 偏置，形状为 (F,)
    - conv_param: 字典，包含以下键:
      - 'stride': 水平和垂直方向上相邻感受野之间的像素数。
      - 'pad': 用于对输入进行零填充的像素数。

    填充时，'pad' 个零应沿输入的高度和宽度轴对称放置（即两侧等量）。
    注意不要直接修改原始输入 x。

    返回一个元组:
    - out: 输出数据，形状为 (N, F, H', W')，其中 H' 和 W' 由下式给出:
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: 实现卷积前向传播。                                                #
    # 提示：你可以使用 np.pad 函数进行填充。                                  #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """卷积层反向传播的朴素实现。

    输入:
    - dout: 上游梯度。
    - cache: (x, w, b, conv_param) 元组，与 conv_forward_naive 中相同

    返回一个元组:
    - dx: 关于 x 的梯度
    - dw: 关于 w 的梯度
    - db: 关于 b 的梯度
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: 实现卷积反向传播。                                                #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """最大池化层前向传播的朴素实现。

    输入:
    - x: 输入数据，形状为 (N, C, H, W)
    - pool_param: 字典，包含以下键:
      - 'pool_height': 每个池化区域的高度
      - 'pool_width': 每个池化区域的宽度
      - 'stride': 相邻池化区域之间的距离

    这里不需要填充，例如你可以假设:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    返回一个元组:
    - out: 输出数据，形状为 (N, C, H', W')，其中 H' 和 W' 由下式给出:
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: 实现最大池化前向传播                                              #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """最大池化层反向传播的朴素实现。

    输入:
    - dout: 上游梯度
    - cache: 前向传播中的 (x, pool_param) 元组。

    返回:
    - dx: 关于 x 的梯度
    """
    dx = None
    ###########################################################################
    # TODO: 实现最大池化反向传播                                              #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """计算空间批归一化的前向传播。

    输入:
    - x: 输入数据，形状为 (N, C, H, W)
    - gamma: 缩放参数，形状为 (C,)
    - beta: 平移参数，形状为 (C,)
    - bn_param: 字典，包含以下键:
      - mode: 'train' 或 'test'；必需
      - eps: 数值稳定性常数
      - momentum: 运行均值/方差的常数。momentum=0 表示在每个时间步完全丢弃
        旧信息，而 momentum=1 表示从不纳入新信息。momentum=0.9 的默认值
        在大多数情况下应该工作良好。
      - running_mean: 形状为 (D,) 的数组，给出特征的运行均值
      - running_var: 形状为 (D,) 的数组，给出特征的运行方差

    返回一个元组:
    - out: 输出数据，形状为 (N, C, H, W)
    - cache: 反向传播所需的值
    """
    out, cache = None, None

    ###########################################################################
    # TODO: 实现空间批归一化的前向传播。                                      #
    #                                                                         #
    # 提示：你可以通过调用上面实现的普通版本的批归一化来实现空间批归一化。    #
    # 你的实现应该非常简短；我们的实现不到五行。                              #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """计算空间批归一化的反向传播。

    输入:
    - dout: 上游梯度，形状为 (N, C, H, W)
    - cache: 前向传播中的值

    返回一个元组:
    - dx: 关于输入的梯度，形状为 (N, C, H, W)
    - dgamma: 关于缩放参数的梯度，形状为 (C,)
    - dbeta: 关于平移参数的梯度，形状为 (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: 实现空间批归一化的反向传播。                                      #
    #                                                                         #
    # 提示：你可以通过调用上面实现的普通版本的批归一化来实现空间批归一化。    #
    # 你的实现应该非常简短；我们的实现不到五行。                              #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """计算空间组归一化的前向传播。

    与层归一化不同，组归一化将数据中的每个条目分成 G 个连续的片段，
    然后独立地对每个片段进行归一化。然后对数据应用逐特征的平移和缩放，
    方式与批归一化和层归一化完全相同。

    输入:
    - x: 输入数据，形状为 (N, C, H, W)
    - gamma: 缩放参数，形状为 (1, C, 1, 1)
    - beta: 平移参数，形状为 (1, C, 1, 1)
    - G: 要分割成的组数，应为 C 的约数
    - gn_param: 字典，包含以下键:
      - eps: 数值稳定性常数

    返回一个元组:
    - out: 输出数据，形状为 (N, C, H, W)
    - cache: 反向传播所需的值
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: 实现空间组归一化的前向传播。                                      #
    # 这将与层归一化的实现非常相似。                                          #
    # 特别是，思考一下如何变换矩阵，使大部分代码与训练时批归一化和层归一化    #
    # 都相似！                                                                #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """计算空间组归一化的反向传播。

    输入:
    - dout: 上游梯度，形状为 (N, C, H, W)
    - cache: 前向传播中的值

    返回一个元组:
    - dx: 关于输入的梯度，形状为 (N, C, H, W)
    - dgamma: 关于缩放参数的梯度，形状为 (1, C, 1, 1)
    - dbeta: 关于平移参数的梯度，形状为 (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: 实现空间组归一化的反向传播。                                      #
    # 这将与层归一化的实现非常相似。                                          #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

import numpy as np

"""
本文件实现了多种常用于训练神经网络的一阶更新规则。
每种更新规则接收当前权重以及损失函数相对于这些权重的梯度，
并生成下一组权重。每种更新规则具有相同的接口：

def update(w, dw, config=None):

输入：
  - w: 表示当前权重的 numpy 数组。
  - dw: 与 w 形状相同的 numpy 数组，表示损失函数相对于 w 的梯度。
  - config: 包含超参数值的字典，例如学习率、动量等。
    如果更新规则需要在多次迭代中缓存值，则 config 也会保存这些缓存值。

返回：
  - next_w: 更新后的下一个点。
  - config: 传递给更新规则下一次迭代的 config 字典。

注意：对于大多数更新规则，默认学习率可能效果不佳；
但其他超参数的默认值应该适用于多种不同的问题。

为了效率，更新规则可能会执行原地更新，直接修改 w 并将 next_w 设为 w。
"""


def sgd(w, dw, config=None):
    """
    执行标准随机梯度下降。

    config 格式：
    - learning_rate: 标量学习率。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    执行带动量的随机梯度下降。

    config 格式：
    - learning_rate: 标量学习率。
    - momentum: 0 到 1 之间的标量，表示动量值。
      将 momentum 设为 0 退化为 sgd。
    - velocity: 与 w 和 dw 形状相同的 numpy 数组，用于存储梯度的移动平均。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: 实现动量更新公式。将更新后的值存储在 next_w 变量中。               #
    # 你还应该使用并更新速度 v。                                               #
    ###########################################################################

    ###########################################################################
    #                             你的代码到此结束                             #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    使用 RMSProp 更新规则，该规则利用梯度平方值的移动平均来设置
    自适应的逐参数学习率。

    config 格式：
    - learning_rate: 标量学习率。
    - decay_rate: 0 到 1 之间的标量，表示平方梯度缓存的衰减率。
    - epsilon: 用于平滑的小标量，以避免除以零。
    - cache: 梯度二阶矩的移动平均。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: 实现 RMSprop 更新公式，将 w 的下一个值存储在 next_w 变量中。      #
    # 不要忘记更新存储在 config['cache'] 中的缓存值。                          #
    ###########################################################################

    ###########################################################################
    #                             你的代码到此结束                             #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    使用 Adam 更新规则，该规则结合了梯度及其平方的移动平均
    以及一个偏差修正项。

    config 格式：
    - learning_rate: 标量学习率。
    - beta1: 梯度一阶矩移动平均的衰减率。
    - beta2: 梯度二阶矩移动平均的衰减率。
    - epsilon: 用于平滑的小标量，以避免除以零。
    - m: 梯度的移动平均。
    - v: 梯度平方的移动平均。
    - t: 迭代次数。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: 实现 Adam 更新公式，将 w 的下一个值存储在 next_w 变量中。         #
    # 不要忘记更新存储在 config 中的 m、v 和 t 变量。                         #
    #                                                                         #
    # 注意：为了匹配参考输出，请在进行任何计算之前先修改 t。                  #
    ###########################################################################
    config['t'] += 1
    t = config['t']
    lr = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['epsilon']
    m = config['m']
    v = config['v']

    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    next_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    config['m'] = m
    config['v'] = v
    ###########################################################################
    #                             你的代码到此结束                             #
    ###########################################################################

    return next_w, config

import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        """模型初始化函数"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练模型"""
        assert x_train.ndim ==1, \
            "简单线性回归模型仅能够处理一维特征向量"
        assert len(x_train) == len(y_train), \
            "特征向量的长度和标签的长度相同"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = (x_train - x_mean).dot(y_train - y_mean)  # 分子
        d = (x_train - x_mean).dot(x_train - x_mean)    # 分母
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "简单线性回归模型仅能够处理一维特征向量"
        assert self.a_ is not None and self.b_ is not None, \
            "先训练之后才能预测"
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x_single，返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        """返回一个可以用来表示对象的可打印字符串"""
        return "SimpleLinearRegression()"




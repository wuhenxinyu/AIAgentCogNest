"""
目标函数最优化问题的工程解法：
    • ALS，Alternating Least Squares，交替最小二乘法
    • SGD，Stochastic Gradient Descent，随机梯度下降

传统推荐方式：
    • 基于内容的推荐
    • 基于协同过滤的推荐
        - 基于用户的协同过滤
        - 基于物品的协同过滤
    • 基于知识的推荐
    • 基于模型的推荐
        - 基于矩阵分解的推荐：
            - 奇异值分解（SVD）
            - 非负矩阵分解（NMF）
            - 概率矩阵分解（PMF）
            - 正则化奇异值分解（SVD++）
            - 时间感知矩阵分解（TAMF）
            - 交替最小二乘法（ALS）矩阵分解
            - 随机梯度下降（SGD）矩阵分解
    • 混合推荐

当前采用矩阵分解的推荐方式：
    - 基于ALS的矩阵分解
"""
# 使用ALS进行矩阵分解
from itertools import product, chain
from copy import deepcopy


"""
此模块定义了一个 `Matrix` 类，用于表示和操作矩阵。该类提供了一系列矩阵基本操作，
包括获取行、列，矩阵转置、求逆、矩阵乘法、标量乘法等，是后续实现 ALS 算法的基础工具类。
"""

class Matrix(object):
    """
    矩阵类，用于表示和操作矩阵。该类封装了矩阵的基本属性和常见操作，
    方便对矩阵进行各种数学运算。
    """
    def __init__(self, data):
        """
        初始化矩阵对象。

        Args:
            data (list): 二维列表，用于表示矩阵的数据。
        """
        self.data = data
        self.shape = (len(data), len(data[0]))

    def row(self, row_no):
        """
        获取矩阵的指定行。

        Args:
            row_no (int): 矩阵的行号，从 0 开始计数。

        Returns:
            Matrix: 一个新的 Matrix 对象，仅包含指定的行。
        """
        return Matrix([self.data[row_no]])

    def col(self, col_no):
        """
        获取矩阵的指定列。

        Args:
            col_no (int): 矩阵的列号，从 0 开始计数。

        Returns:
            Matrix: 一个新的 Matrix 对象，仅包含指定的列。
        """
        m = self.shape[0]
        return Matrix([[self.data[i][col_no]] for i in range(m)])

    @property
    def is_square(self):
        """
        检查矩阵是否为方阵（行数和列数相等）。

        Returns:
            bool: 如果是方阵返回 True，否则返回 False。
        """
        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        """
        求原矩阵的转置矩阵。

        Returns:
            Matrix: 一个新的 Matrix 对象，表示原矩阵的转置矩阵。
        """
        data = list(map(list, zip(*self.data)))
        return Matrix(data)

    def _eye(self, n):
        """
        获取指定阶数的单位矩阵（私有方法）。

        Args:
            n (int): 单位矩阵的阶数。

        Returns:
            list: 二维列表，表示指定阶数的单位矩阵。
        """
        return [[0 if i != j else 1 for j in range(n)] for i in range(n)]

    @property
    def eye(self):
        """
        获取与当前矩阵同阶的单位矩阵。

        Raises:
            AssertionError: 如果当前矩阵不是方阵，抛出此异常。

        Returns:
            Matrix: 一个新的 Matrix 对象，表示与当前矩阵同阶的单位矩阵。
        """
        assert self.is_square, "The matrix has to be square!"
        data = self._eye(self.shape[0])
        return Matrix(data)

    def _gaussian_elimination(self, aug_matrix):
        """
        对增广矩阵进行高斯消元，将增广矩阵左侧的方阵化简为单位对角矩阵（私有方法）。

        Args:
            aug_matrix (list): 二维列表，包含整数或浮点数，表示增广矩阵。

        Returns:
            list: 二维列表，包含整数或浮点数，表示消元后的增广矩阵。
        """
        n = len(aug_matrix)
        m = len(aug_matrix[0])

        # 正向消元，从顶部到底部
        for col_idx in range(n):
            # 检查对角线上的元素是否为零
            if aug_matrix[col_idx][col_idx] == 0:
                row_idx = col_idx
                # 找到同一列上元素不为零的行
                while row_idx < n and aug_matrix[row_idx][col_idx] == 0:
                    row_idx += 1
                # 将找到的行加到对角元素所在行
                for i in range(col_idx, m):
                    aug_matrix[col_idx][i] += aug_matrix[row_idx][i]

            # 消去对角元素下方的非零元素
            for i in range(col_idx + 1, n):
                # 跳过零元素
                if aug_matrix[i][col_idx] == 0:
                    continue
                # 计算消元系数
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                # 消去非零元素
                for j in range(col_idx, m):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # 反向消元，从底部到顶部
        for col_idx in range(n - 1, -1, -1):
            # 消去对角元素上方的非零元素
            for i in range(col_idx):
                # 跳过零元素
                if aug_matrix[i][col_idx] == 0:
                    continue
                # 计算消元系数
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                # 消去非零元素
                for j in chain(range(i, col_idx + 1), range(n, m)):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # 将对角元素归一化为 1
        for i in range(n):
            k = 1 / aug_matrix[i][i]
            aug_matrix[i][i] *= k
            for j in range(n, m):
                aug_matrix[i][j] *= k

        return aug_matrix

    def _inverse(self, data):
        """
        求矩阵的逆矩阵（私有方法）。

        Args:
            data (list): 二维列表，包含整数或浮点数，表示待求逆的矩阵。

        Returns:
            list: 二维列表，包含整数或浮点数，表示逆矩阵。
        """
        n = len(data)
        unit_matrix = self._eye(n)
        # 构造增广矩阵 [A | I]
        aug_matrix = [a + b for a, b in zip(self.data, unit_matrix)]
        # 进行高斯消元
        ret = self._gaussian_elimination(aug_matrix)
        # 提取增广矩阵右侧的部分，即为逆矩阵
        return list(map(lambda x: x[n:], ret))

    @property
    def inverse(self):
        """
        求当前矩阵的逆矩阵。

        Raises:
            AssertionError: 如果当前矩阵不是方阵，抛出此异常。

        Returns:
            Matrix: 一个新的 Matrix 对象，表示当前矩阵的逆矩阵。
        """
        assert self.is_square, "The matrix has to be square!"
        data = self._inverse(self.data)
        return Matrix(data)

    def _row_mul(self, row_A, row_B):
        """
        将两个一维数组中相同下标的元素相乘并求和（私有方法）。

        Args:
            row_A (list): 一维列表，包含浮点数或整数。
            row_B (list): 一维列表，包含浮点数或整数。

        Returns:
            float or int: 相乘求和的结果。
        """
        return sum(x[0] * x[1] for x in zip(row_A, row_B))

    def _mat_mul(self, row_A, B):
        """
        矩阵乘法的辅助函数（私有方法），用于计算一行与另一个矩阵的乘积。

        Args:
            row_A (list): 一维列表，包含浮点数或整数，表示矩阵的一行。
            B (Matrix): 另一个 Matrix 对象。

        Returns:
            list: 一维列表，包含浮点数或整数，表示乘积结果的一行。
        """
        row_pairs = product([row_A], B.transpose.data)
        return [self._row_mul(*row_pair) for row_pair in row_pairs]

    def mat_mul(self, B):
        """
        矩阵乘法，计算当前矩阵与另一个矩阵的乘积。

        Args:
            B (Matrix): 另一个 Matrix 对象。

        Raises:
            AssertionError: 如果当前矩阵的列数与矩阵 B 的行数不相等，抛出此异常。

        Returns:
            Matrix: 一个新的 Matrix 对象，表示两个矩阵的乘积。
        """
        error_msg = "A's column count does not match B's row count!"
        assert self.shape[1] == B.shape[0], error_msg
        return Matrix([self._mat_mul(row_A, B) for row_A in self.data])

    def _mean(self, data):
        """
        计算矩阵中所有样本的平均值（按列计算）（私有方法）。

        Args:
            X (list): 二维列表，包含整数或浮点数，表示矩阵数据。

        Returns:
            list: 一维列表，包含整数或浮点数，表示每列的平均值。
        """
        m = len(data)
        n = len(data[0])
        ret = [0 for _ in range(n)]
        for row in data:
            for j in range(n):
                ret[j] += row[j] / m
        return ret

    def mean(self):
        """
        计算当前矩阵中所有样本的平均值（按列计算）。

        Returns:
            Matrix: 一个新的 Matrix 对象，表示每列的平均值。
        """
        return Matrix(self._mean(self.data))

    def scala_mul(self, scala):
        """
        矩阵的标量乘法，将矩阵的每个元素乘以指定的标量。

        Args:
            scala (float): 标量值。

        Returns:
            Matrix: 一个新的 Matrix 对象，表示标量乘法的结果。
        """
        m, n = self.shape
        data = deepcopy(self.data)
        for i in range(m):
            for j in range(n):
                data[i][j] *= scala
        return Matrix(data)


import pandas as pd
import numpy as np
import random
from collections import defaultdict

class ALS(object):
    """
    该类实现了交替最小二乘法（Alternating Least Squares, ALS）用于矩阵分解。
    矩阵分解常用于推荐系统，将用户-物品评分矩阵分解为用户矩阵和物品矩阵，
    通过迭代优化这两个矩阵来逼近原始评分矩阵，进而实现推荐功能。
    """
    def __init__(self):
        """
        初始化ALS类的实例，设置类的属性为None，这些属性将在数据处理和模型训练过程中被赋值。
        
        属性说明:
            user_ids (tuple): 所有用户ID的元组。
            item_ids (tuple): 所有物品ID的元组。
            user_ids_dict (dict): 用户ID到索引的映射字典。
            item_ids_dict (dict): 物品ID到索引的映射字典。
            user_matrix (Matrix): 用户矩阵，形状为 k * m，m 为用户数量。
            item_matrix (Matrix): 物品矩阵，形状为 k * n，n 为物品数量。
            user_items (dict): 用户已评分的物品集合，键为用户ID，值为物品ID集合。
            shape (tuple): 评分矩阵的形状，(用户数量, 物品数量)。
            rmse (float): 均方根误差，用于评估模型的预测效果。
        """
        self.user_ids = None
        self.item_ids = None
        self.user_ids_dict = None
        self.item_ids_dict = None
        self.user_matrix = None
        self.item_matrix = None
        self.user_items = None
        self.shape = None
        self.rmse = None


    def _process_data(self, X):
        """
        将评分矩阵X转化为稀疏矩阵，同时初始化用户和物品的ID及其映射关系。
        
        输入参数:
            X (list): 二维列表，每个子列表包含三个元素 [user_id, item_id, rating]，
                     分别表示用户ID、物品ID和评分。
        
        输出结果:
            dict: 稀疏评分矩阵，格式为 {user_id: {item_id: rating}}。
            dict: 转置后的稀疏评分矩阵，格式为 {item_id: {user_id: rating}}。
        """
        # 获取所有用户ID并去重，转换为元组
        self.user_ids = tuple((set(map(lambda x: x[0], X))))
        # 创建用户ID到索引的映射字典
        self.user_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.user_ids)))
     
        # 获取所有物品ID并去重，转换为元组
        self.item_ids = tuple((set(map(lambda x: x[1], X))))
        # 创建物品ID到索引的映射字典
        self.item_ids_dict = dict(map(lambda x: x[::-1], enumerate(self.item_ids)))
     
        # 记录评分矩阵的形状 (用户数量, 物品数量)
        self.shape = (len(self.user_ids), len(self.item_ids))
     
        # 初始化两个默认字典，用于存储稀疏评分矩阵及其转置
        ratings = defaultdict(lambda: defaultdict(int))
        ratings_T = defaultdict(lambda: defaultdict(int))
        # 遍历评分数据，填充稀疏矩阵
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating
     
        # 检查用户ID数量和评分矩阵中的用户数量是否一致
        err_msg = "Length of user_ids %d and ratings %d not match!" % (len(self.user_ids), len(ratings))
        assert len(self.user_ids) == len(ratings), err_msg
     
        # 检查物品ID数量和转置评分矩阵中的物品数量是否一致
        err_msg = "Length of item_ids %d and ratings_T %d not match!" % (len(self.item_ids), len(ratings_T))
        assert len(self.item_ids) == len(ratings_T), err_msg
        return ratings, ratings_T

     
    def _users_mul_ratings(self, users, ratings_T):
        """
        实现用户矩阵(稠密) 与 评分矩阵（稀疏）相乘的操作。
        
        参数说明:
            users (Matrix): k * m 矩阵，m 表示用户数量，k 表示隐特征数量。
            ratings_T (dict): 物品被用户评分的稀疏矩阵，格式为 {item_id: {user_id: rating}}。
        
        返回值:
            Matrix: 物品矩阵，形状为 k * n，n 表示物品数量。
        """
        def f(users_row, item_id):
            """
            辅助函数，计算用户矩阵的一行与某个物品的评分向量的乘积。
            
            参数说明:
                users_row (list): 用户矩阵的一行。
                item_id (int): 物品ID。
            
            返回值:
                float: 计算结果。
            """
            # 获取对该物品评分的用户ID
            user_ids = iter(ratings_T[item_id].keys())
            # 获取对应的评分
            scores = iter(ratings_T[item_id].values())
            # 将用户ID转换为用户矩阵中的列索引
            col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
            # 获取用户矩阵中对应列的值
            _users_row = map(lambda x: users_row[x], col_nos)
            # 对应元素相乘并求和
            return sum(a * b for a, b in zip(_users_row, scores))
     
        # 对用户矩阵的每一行，计算与所有物品的乘积
        ret = [[f(users_row, item_id) for item_id in self.item_ids] for users_row in users.data]
        return Matrix(ret)

    def _items_mul_ratings(self, items, ratings):
        """
        实现物品矩阵（稠密）与评分矩阵（稀疏）相乘的操作。
        
        参数说明:
            items (Matrix): k * n 矩阵，n 表示物品数量，k 表示隐特征数量。
            ratings (dict): 用户对物品评分的稀疏矩阵，格式为 {user_id: {item_id: rating}}。
        
        返回值:
            Matrix: 用户矩阵，形状为 k * m，m 表示用户数量。
        """
        def f(items_row, user_id):
            """
            辅助函数，计算物品矩阵的一行与某个用户的评分向量的乘积。
            
            参数说明:
                items_row (list): 物品矩阵的一行。
                user_id (int): 用户ID。
            
            返回值:
                float: 计算结果。
            """
            # 获取该用户评分的物品ID
            item_ids = iter(ratings[user_id].keys())
            # 获取对应的评分
            scores = iter(ratings[user_id].values())
            # 将物品ID转换为物品矩阵中的列索引
            col_nos = map(lambda x: self.item_ids_dict[x], item_ids)
            # 获取物品矩阵中对应列的值
            _items_row = map(lambda x: items_row[x], col_nos)
            # 对应元素相乘并求和
            return sum(a * b for a, b in zip(_items_row, scores))
     
        # 对物品矩阵的每一行，计算与所有用户的乘积
        ret = [[f(items_row, user_id) for user_id in self.user_ids] for items_row in items.data]
        return Matrix(ret)

    # 生成随机矩阵
    def _gen_random_matrix(self, n_rows, n_colums):
        """
        生成指定行数和列数的随机矩阵，矩阵元素取值范围为 [0, 1)。
        
        参数说明:
            n_rows (int): 矩阵的行数。
            n_colums (int): 矩阵的列数。
        
        返回值:
            Matrix: 生成的随机矩阵对象。
        """
        #print(n_colums, ' ', n_rows)
        #data = [[random() for _ in range(n_colums)] for _ in range(n_rows)]
        #d = 2
        data = np.random.rand(n_rows, n_colums)
        return Matrix(data)


    # 计算RMSE
    def _get_rmse(self, ratings):
        """
        计算模型预测评分与真实评分之间的均方根误差（RMSE）。
        
        参数说明:
            ratings (dict): 用户对物品评分的稀疏矩阵，格式为 {user_id: {item_id: rating}}。
        
        返回值:
            float: 均方根误差。
        """
        m, n = self.shape
        mse = 0.0
        # 计算有评分的元素数量
        n_elements = sum(map(len, ratings.values()))
        # 遍历所有用户和物品
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                item_id = self.item_ids[j]
                rating = ratings[user_id][item_id]
                # 只对有真实评分的元素计算误差
                if rating > 0:
                    # 获取用户矩阵中该用户对应的行
                    user_row = self.user_matrix.col(i).transpose
                    # 获取物品矩阵中该物品对应的列
                    item_col = self.item_matrix.col(j)
                    # 计算预测评分
                    rating_hat = user_row.mat_mul(item_col).data[0][0]
                    # 计算平方误差
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error / n_elements
        # 返回均方根误差
        return mse ** 0.5

    # 模型训练
    def fit(self, X, k, max_iter=10):
        """
        使用交替最小二乘法训练模型，迭代优化用户矩阵和物品矩阵。
        
        参数说明:
            X (list): 二维列表，每个子列表包含三个元素 [user_id, item_id, rating]。
            k (int): 隐特征数量，需小于评分矩阵的最小秩。
            max_iter (int, optional): 最大迭代次数，默认为10。
        """
        # 处理输入数据，得到稀疏评分矩阵及其转置
        ratings, ratings_T = self._process_data(X)
        # 记录每个用户已评分的物品集合
        self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
        m, n = self.shape
     
        # 检查隐特征数量是否小于评分矩阵的最小秩
        error_msg = "Parameter k must be less than the rank of original matrix"
        assert k < min(m, n), error_msg
     
        # 初始化用户矩阵为随机矩阵
        self.user_matrix = self._gen_random_matrix(k, m)
     
        # 交替迭代优化用户矩阵和物品矩阵
        for i in range(max_iter):
            if i % 2:
                # 奇数迭代，固定物品矩阵，更新用户矩阵
                items = self.item_matrix
                self.user_matrix = self._items_mul_ratings(
                    items.mat_mul(items.transpose).inverse.mat_mul(items),
                    ratings
                )
            else:
                # 偶数迭代，固定用户矩阵，更新物品矩阵
                users = self.user_matrix
                self.item_matrix = self._users_mul_ratings(
                    users.mat_mul(users.transpose).inverse.mat_mul(users),
                    ratings_T
                )
            # 计算当前迭代的RMSE
            rmse = self._get_rmse(ratings)
            print("Iterations: %d, RMSE: %.6f" % (i + 1, rmse))
     
        # 记录最终的RMSE
        self.rmse = rmse

    # Top-n推荐，用户列表：user_id, n_items: Top-n
    def _predict(self, user_id, n_items):
        """
        为单个用户生成Top-n推荐物品列表。
        
        参数说明:
            user_id (int): 用户ID。
            n_items (int): 推荐的物品数量。
        
        返回值:
            list: 推荐的物品列表，每个元素为 (item_id, score) 元组，按评分降序排列。
        """
        # 获取用户矩阵中该用户对应的列，并转置
        users_col = self.user_matrix.col(self.user_ids_dict[user_id])
        users_col = users_col.transpose
     
        # 计算该用户对所有物品的预测评分
        items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
        items_scores = map(lambda x: (self.item_ids[x[0]], x[1]), items_col)
        # 过滤掉用户已经评分过的物品
        viewed_items = self.user_items[user_id]
        items_scores = filter(lambda x: x[0] not in viewed_items, items_scores)
     
        # 按预测评分降序排序，取前n_items个物品
        return sorted(items_scores, key=lambda x: x[1], reverse=True)[:n_items]

    # 预测多个用户
    def predict(self, user_ids, n_items=10):
        """
        为多个用户生成Top-n推荐物品列表。
        
        参数说明:
            user_ids (list): 用户ID列表。
            n_items (int, optional): 推荐的物品数量，默认为10。
        
        返回值:
            list: 每个用户的推荐物品列表，每个元素为单个用户的推荐结果。
        """
        return [self._predict(user_id, n_items) for user_id in user_ids]

# 以下是自己写的部分
# 此段代码的主要作用是演示如何使用ALS算法进行矩阵分解，并基于分解结果为用户进行Top-N推荐。
# 具体流程包括数据加载、模型训练和推荐结果展示。
# 定义格式化预测结果的函数，将物品ID和评分格式化为指定字符串
def format_prediction(item_id, score):
    return "item_id:%d score:%.2f" % (item_id, score)

# 定义加载电影评分数据的函数，从指定文件中读取评分数据
def load_movie_ratings(file_name):
    # 打开指定文件
    f = open(file_name)
    # 将文件内容转换为迭代器
    lines = iter(f)
    # 获取第一行的列名，并去除最后一个换行符，按逗号分割后取前几个元素，用逗号连接
    col_names = ", ".join(next(lines)[:-1].split(",")[:-1])
    # 打印列名
    print("The column names are: %s." % col_names)
    # 逐行读取文件内容，将每行数据按逗号分割并转换为对应类型（第三个元素为浮点数，其余为整数）
    data = [[float(x) if i == 2 else int(x)
             for i, x in enumerate(line[:-1].split(",")[:-1])]
            for line in lines]
    # 关闭文件
    f.close()

    return data

# 打印提示信息，表示即将使用ALS算法
print("使用ALS算法") 
# 初始化ALS模型实例
model = ALS()
# 调用数据加载函数，从指定文件中加载电影评分数据
X = load_movie_ratings('21-Fine-tuning微调艺术/MovieLens/ratings_small.csv')
# 注释掉的打印语句，可用于查看加载的数据
#print(X)
# 调用模型的fit方法进行训练，设置隐特征数量k为3，最大迭代次数为2
# 此处的k并非聚类个数，而是矩阵分解中的隐特征数量
model.fit(X, k=3, max_iter=2)

# 以下是一段被注释的代码，展示了使用另一组示例数据进行模型训练
"""
# 定义示例评分数据，每个子列表表示 [user_id, item_id, rating]
X = np.array([[1,1,1], [1,2,1], [2,1,1], [2,3,1], [3,2,1], [3,3,1], [4,1,1], [4,2,1],
              [5,4,1], [5,5,1], [6,4,1], [6,6,1], [7,5,1], [7,6,1], [8,4,1], [8,5,1], [9,4,1], [9,5,1],
              [10,7,1], [10,8,1], [11,8,1], [11,9,1], [12,7,1], [12,9,1]])
# 调用模型的fit方法进行训练，设置隐特征数量k为3，最大迭代次数为20
model.fit(X, k=3, max_iter=20)
"""

# 打印提示信息，表示即将为用户进行推荐
print("对用户进行推荐")
# 定义要进行推荐的用户ID列表，范围从1到12
user_ids = range(1, 13)
# 调用模型的predict方法，为用户列表中的每个用户进行Top-2推荐
# 注意：原代码中model.predict方法可能未在提供的代码中定义，需确保该方法存在
predictions = model.predict(user_ids, n_items=2)
# 打印预测结果
print(predictions)
# 遍历用户ID和对应的预测结果
for user_id, prediction in zip(user_ids, predictions):
    # 对每个预测结果调用format_prediction函数进行格式化
    _prediction = [format_prediction(item_id, score) for item_id, score in prediction]
    # 打印每个用户的推荐结果
    print("User id:%d recommedation: %s" % (user_id, _prediction))

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
from utils.monitor import resource_monitor_decorator
from loguru import logger
import sys

device_name = os.getenv('DEVICE') or 'GPU:0'


def compute_knn(
    X,
    y,
    k,
    minority_class=1,
    batch_size=100,
):
    minority_samples = tf.squeeze(tf.gather(X, tf.where(y == minority_class)),
                                  axis=1)
    # 选择少数类样本
    num_samples = X.shape[0]
    num_minority_samples = len(minority_samples)
    # 初始化距离矩阵
    total_distances = np.zeros((num_minority_samples, num_samples),
                               dtype=np.float32)
    # 分片
    for start in tqdm(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)
        batch_X = X[start:end]
        # 计算少数类样本到当前批次样本的距离
        distances = tf.norm(tf.expand_dims(minority_samples, axis=1) -
                            tf.expand_dims(batch_X, axis=0),
                            axis=-1)

        # 将当前批次的距离填入总距离矩阵
        total_distances[:, start:end] = distances.numpy()
    logger.info(total_distances.shape)
    logger.info('sorting by cpu...')
    # 获取KNN索引
    knn_indices = np.argsort(total_distances, axis=-1)[:, :k]
    logger.info('sorting by cpu, done')
    return knn_indices


def generate_synthetic_samples(
    X,
    y,
    knn_indices,
    k_neighbors=5,
    minority_class=1,
    ratio=1.0,
):
    # 找到少数类样本
    minority_samples = tf.squeeze(tf.gather(X, tf.where(y == minority_class)),
                                  axis=1)
    # 对于每个少数样本：
    # 计算少数类样本的K近邻
    # 标记边界样本
    # 生成合成样本

    # TODO 加速 https://tensorflow.google.cn/guide/random_numbers?hl=zh-cn
    rng = tf.random.Generator.from_seed(114514)
    random_ij = rng.uniform(shape=(minority_samples.shape[0], int(ratio)),
                            minval=0,
                            maxval=k_neighbors)  # neighbor index
    random_ratio = rng.uniform(shape=(minority_samples.shape[0], int(ratio)),
                               minval=0,
                               maxval=1)  # synthetic = p + r * dist(p,p_i)
    # 计算邻居的标签
    neighbors_label = tf.gather(y, knn_indices)
    # 计算少数类样本的k近邻中正类样本的数量
    num_majority_neighbors = tf.reduce_sum(tf.cast(
        neighbors_label != minority_class, tf.int32),
                                           axis=1)
    # 找出危险样本
    danger_mask = tf.logical_and(
        num_majority_neighbors > tf.cast(k_neighbors / 2, tf.int32),
        num_majority_neighbors < tf.cast(k_neighbors, tf.int32))
    danger_indices = tf.where(danger_mask)[:, 0]
    # 获取危险样本及其对应的近邻索引
    danger_samples = tf.gather(minority_samples, danger_indices)
    danger_knn_indices = tf.gather(knn_indices, danger_indices)
    danger_random_ij = tf.gather(random_ij, danger_indices)
    danger_random_ratio = tf.gather(random_ratio, danger_indices)
    # 将所有索引转换为整数类型
    danger_knn_indices = tf.cast(danger_knn_indices, tf.int32)
    danger_random_ij = tf.cast(danger_random_ij, tf.int32)
    # 计算差异向量
    knn_selected_indices = tf.gather(danger_knn_indices,
                                     danger_random_ij,
                                     batch_dims=1)
    diff = tf.gather(X, knn_selected_indices) - tf.expand_dims(danger_samples,
                                                               axis=1)
    # 计算合成样本
    new_samples = tf.expand_dims(
        danger_samples,
        axis=1) + tf.expand_dims(danger_random_ratio, axis=2) * diff
    # 展平结果
    synthetic_samples = tf.reshape(new_samples, [-1, X.shape[1]])
    return tf.convert_to_tensor(synthetic_samples, dtype=X.dtype)


@resource_monitor_decorator
def borderline_smote(X, y, n_samples, minority_class=1, k_neighbors=5):
    with tf.device(f'/{device_name}'):
        knn_indices = compute_knn(X, y, k_neighbors, minority_class)
        synthetic_samples = generate_synthetic_samples(
            X,
            y,
            knn_indices,
            k_neighbors=k_neighbors,
            minority_class=minority_class,
            ratio=n_samples,
        )
    return tf.concat([X, synthetic_samples], axis=0), tf.concat(
        [
            y,
            tf.broadcast_to(
                [minority_class],
                shape=[synthetic_samples.shape[0]],
            ),
        ],
        axis=0,
    )


def _test():
    N = 5000
    P = 50
    num_samples = P + N
    num_features = 24 * 24
    X_large = tf.random.normal((num_samples, num_features))
    y_large = tf.concat(
        [tf.zeros(N, dtype=tf.int32),
         tf.ones(P, dtype=tf.int32)], axis=0)
    minority_class = 1
    k_neighbors = 5
    n_samples_per_minority = N / P
    # time1 = time.time()
    synthetic_samples_large, y = borderline_smote(
        X_large,
        y_large,
        n_samples_per_minority,
        minority_class,
        k_neighbors,
    )
    # logger.info('data augmentation cost time (s^-1) :', time.time() - time1)
    # 恢复形状
    original_shape = (-1, 24, 24)
    # synthetic = tf.concat([X_large, synthetic_samples_large], axis=0)
    synthetic = tf.reshape(
        synthetic_samples_large,
        original_shape,
    )

    logger.info(synthetic.shape)
    logger.info(y.shape)
    # logger.info(synthetic_samples_large.shape)
    # return synthetic_samples_large


if __name__ == "__main__":
    _test()

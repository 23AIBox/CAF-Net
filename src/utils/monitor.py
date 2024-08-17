import psutil
import time
import os
import tensorflow as tf
from loguru import logger
import sys

device_name = os.getenv('DEVICE') or 'GPU:0'


def monitor(func, name: str = 'default', *args, **kwargs):
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024**2)
    start_gpu_memory = tf.config.experimental.get_memory_info(
        f'{device_name}')['current'] / (1024**2)
    result = func(*args, **kwargs)
    end_time = time.time()
    end_memory = process.memory_info().rss / (1024**2)
    end_gpu_memory = tf.config.experimental.get_memory_info(
        f'{device_name}')['current'] / (1024**2)
    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory
    gpu_memory_used = end_gpu_memory - start_gpu_memory
    logger.warning(f"[{name}]")
    logger.warning(f"Time elapsed: {elapsed_time:.2f} seconds")
    logger.info(f"Memory used: {memory_used:.2f} MB")
    logger.info(f"GPU memory used: {gpu_memory_used:.2f} MB")
    return result


def resource_monitor_decorator(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024**2)
        start_gpu_memory = tf.config.experimental.get_memory_info(
            f'{device_name}')['current'] / (1024**2)
        result = func(*args, **kwargs)
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024**2)
        end_gpu_memory = tf.config.experimental.get_memory_info(
            f'{device_name}')['current'] / (1024**2)
        elapsed_time = end_time - start_time
        memory_used = end_memory - start_memory
        gpu_memory_used = end_gpu_memory - start_gpu_memory
        logger.warning(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"Memory used: {memory_used:.2f} MB")
        logger.info(f"GPU memory used: {gpu_memory_used:.2f} MB")
        return result

    return wrapper

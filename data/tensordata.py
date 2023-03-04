
import torch
import numpy as np

from sequence_payload.TrafficLog.setLog import logger


def get_tensor_data(pcap_file, seq_file, label_file, trimed_file_len):
    # 载入 npy 数据
    pcap_data = np.load(pcap_file)  # 获得 pcap 文件
    seq_data = np.load(seq_file)
    label_data = np.load(label_file)  # 获得 label 数据

    # 将 npy 数据转换为 tensor 数据
    pcap_data = torch.from_numpy(pcap_data.reshape(-1, 1, trimed_file_len)).float()
    seq_data = torch.from_numpy(seq_data).float()
    label_data = torch.from_numpy(label_data).long()
    logger.info('pcap 文件大小, {}; seq文件大小:{}; label 文件大小: {}'.format(pcap_data.shape, seq_data.shape,
                                                                               label_data.shape))

    return pcap_data, seq_data, label_data
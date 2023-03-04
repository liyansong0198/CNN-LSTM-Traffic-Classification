import torch
import numpy as np

from sequence_payload.TrafficLog.setLog import logger


def data_loader(pcap_file,seq_file, label_file, trimed_file_len, batch_size=256, workers=1, pin_memory=True):
    # 载入 npy 数据
    pcap_data = np.load(pcap_file) # 获得 pcap 文件
    seq_data = np.load(seq_file)
    label_data = np.load(label_file) # 获得 label 数据

    # 将 npy 数据转换为 tensor 数据
    pcap_data = torch.from_numpy(pcap_data.reshape(-1, 1, trimed_file_len)).float()
    seq_data = torch.from_numpy(seq_data).float()
    label_data = torch.from_numpy(label_data).long()
    logger.info('pcap 文件大小, {}; seq文件大小:{}; label 文件大小: {}'.format(pcap_data.shape, seq_data.shape, label_data.shape))
    
    # 将 tensor 数据转换为 Dataset->Dataloader
    res_dataset = torch.utils.data.TensorDataset(pcap_data,seq_data, label_data) # 合并数据
    res_dataloader = torch.utils.data.DataLoader(
        dataset=res_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=1        # set multi-work num read data
    )

    return res_dataloader
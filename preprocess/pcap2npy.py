
import os
import json
import binascii
import numpy as np
from self.preprocess.splitTrain import get_file_path
from self.TrafficLog.setLog import logger

def save_pcap2npy(fold_path,file_name, npy_data, label2index = {}):
    index = 0
    key = fold_path.split("\\")[-1]
    pcap_dict = get_file_path(fold_path)[key]
    for label, app_npy_list in pcap_dict.items():
        if label not in label2index:
            label2index[label] = index
            index = index + 1
        assert "payload" in app_npy_list[0]
        ip_lengths = np.load(app_npy_list[0])
        y = np.array([label2index[label] for i in range(ip_lengths.shape[0])])
        np.save(os.path.join(fold_path,label, 'label.npy'), y)
    logger.info("标签数据生成完毕")

    return label2index
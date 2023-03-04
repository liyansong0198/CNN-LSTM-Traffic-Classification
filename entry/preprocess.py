
import os
import shutil


from sequence_payload.TrafficLog.setLog import logger
from sequence_payload.utils.setConfig import setup_config

from sequence_payload.preprocess.pcap2session import getPcapLength_Payload
from sequence_payload.preprocess.splitTrain import get_train_test,get_one_npy,merge
from sequence_payload.preprocess.pcap2npy import save_pcap2npy




def preprocess_pipeline():
    """对流量进行预处理, 处理流程为:

    1. 接着将 pcapng 文件转换为 pcap 文件


    6. 对于每一类的文件, 划分训练集和测试集, 获得每一类的所有的 pcap 的路径
    7. 将所有的文件, 最终保存为 npy 的格式
    """
    cfg = setup_config() # 获取 config 文件
    logger.info(cfg)

    # 提取seq pay iat
    getPcapLength_Payload(cfg.pcap_path.traffic_path,cfg.pcap_path.new_pcap_path,cfg.preprocess.packet_num,cfg.preprocess.byte_num,cfg.preprocess.seq_len) # 将 pcap 转换为 session
    # 合并同一应用的所有npy文件
    get_one_npy(cfg.pcap_path.new_pcap_path)
    # 生成标签
    label2index = save_pcap2npy(cfg.pcap_path.new_pcap_path, 'train', "cfg.pcap_path.statistic_feature") # 保存 train 的 npy 文件
    # 标签合并
    merge(folder_path=cfg.pcap_path.new_pcap_path,npy_path=cfg.pcap_path.npy_path)
    # 数据集划分
    get_train_test(npy_path=cfg.pcap_path.npy_path,train_size=0.8)
    logger.info('index 与 label 的关系, {}'.format(label2index))
    print(label2index)

if __name__ == "__main__":
    preprocess_pipeline()
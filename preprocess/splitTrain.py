
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sequence_payload.TrafficLog.setLog import logger

import shutil

def get_file_path(folder_path):
    """获得 folder_path 下 pcap 文件的路径, 以 dict 的形式返回. 
    返回的包含每个大类(Chat, Email), 下每个小类(AIMchat1, aim_chat_3a), 中 pcap 的文件路径.
    返回数据类型如下所示:
    {
        'Chat': {
            'AIMchat1': ['D:\\Traffic-Classification\\data\\preprocess_data\\Chat\\AIMchat1\\AIMchat1.pcap.TCP_131-202-240-87_13393_178-237-24-202_443.pcap', ...]
            'aim_chat_3a': [...],
            ...
        },
        'Email': {
            'email1a': [],
            ...
        },
        ...
    }

    Args:
        folder_path (str): 包含 pcap 文件的根目录名称
    """
    pcap_dict = {}
    for (root, _, files) in os.walk(folder_path):
        if len(files) > 0:
            logger.info('正在记录 {} 下的 pcap 文件'.format(root))
            folder_name_list = os.path.normpath(root).split(os.sep) # 将 'D:\Traffic-Classification\data\preprocess_data' 返回为列表 ['D:', 'Traffic-Classification', 'data', 'preprocess_data']
            top_category, second_category = folder_name_list[-2], folder_name_list[-1]
            if top_category not in pcap_dict:
                pcap_dict[top_category] = {}
            if second_category not in pcap_dict[top_category]:
                pcap_dict[top_category][second_category] = []
            for Ufile in files:
                pcapPath = os.path.join(root, Ufile) # 需要转换的pcap文件的完整路径
                pcap_dict[top_category][second_category].append(pcapPath)
    logger.info('将所有的 pcap 文件整理为 dict !')
    logger.info('==========\n')
    return pcap_dict

def get_index(app_npy_list,tailor):
    for i,app_str in enumerate(app_npy_list):
        if app_str.endswith(tailor):
            return i
    return -1

def get_one_npy(folder_path):
    # 合并
    pcap_dict = get_file_path(folder_path=folder_path)
    for app,app_npy_dict in pcap_dict.items():
        payload_ss = []
        ip_seq_ss = []
        for remove_name,app_npy_list in app_npy_dict.items():

            pay_index = get_index(app_npy_list,tailor="payload.npy")
            seq_index = get_index(app_npy_list, tailor="ip_lengths.npy")

            payload_s = np.load(app_npy_list[pay_index])
            ip_seq_s = np.load(app_npy_list[seq_index])
            payload_ss.extend(payload_s)
            ip_seq_ss.extend(ip_seq_s)

            remove_dir = os.path.join(folder_path,app,remove_name)
            shutil.rmtree(remove_dir)
        payload_ss = np.array(payload_ss)
        ip_seq_ss = np.array(ip_seq_ss)

        np.save(os.path.join(folder_path, app,'payload.npy'), payload_ss)
        np.save(os.path.join(folder_path,app,"ip_lengths.npy"),ip_seq_ss)
        logger.info('完成 {} 文件的合并.'.format(app))
    # logger.info('处理完成文件 {}'.format(Ufile))
    logger.info('完成 npy 文件的合并.')
    logger.info('============\n')


def get_index(app_npy_list,tailor):
    for i,app_str in enumerate(app_npy_list):
        if app_str.endswith(tailor):
            return i
    return -1

def merge(folder_path,npy_path):
    # 合并
    key = folder_path.split("\\")[-1]
    pcap_dict = get_file_path(folder_path)[key]
    seq = []
    pay=[]

    y_ss = []
    for app, app_npy_list in pcap_dict.items():
        pay_index= get_index(app_npy_list,"payload.npy")
        seq_index= get_index(app_npy_list,"ip_lengths.npy")
        label_index= get_index(app_npy_list,"label.npy")

        payload = np.load(app_npy_list[pay_index])
        ip_lengths = np.load(app_npy_list[seq_index])
        y_s = np.load(app_npy_list[label_index])

        pay.extend(payload)
        seq.extend(ip_lengths)
        y_ss.extend(y_s)


    state = np.random.get_state()
    np.random.shuffle(seq)

    np.random.set_state(state)
    np.random.shuffle(pay)

    np.random.set_state(state)
    np.random.shuffle(y_ss)


    pay = np.array(pay)
    seq = np.array(seq)
    y_ss = np.array(y_ss)

    os.makedirs(npy_path, exist_ok=True)

    np.save(os.path.join(npy_path, 'pay_load.npy'), pay)
    np.save(os.path.join(npy_path,'ip_lengths.npy'),seq)
    np.save(os.path.join(npy_path,'label.npy'),y_ss)
    logger.info('完成 npy 文件的合并.')
    logger.info('============\n')


def get_train_test(npy_path, train_size):
    app_npy_list = os.listdir(npy_path)

    seq_index = get_index(app_npy_list,"ip_lengths.npy")
    ip_lengths = np.load(os.path.join(npy_path,app_npy_list[seq_index]))

    pay_index = get_index(app_npy_list,"payload.npy")
    payload = np.load(os.path.join(npy_path,app_npy_list[pay_index]))

    label_index = get_index(app_npy_list,"label.npy")
    label = np.load(os.path.join(npy_path,app_npy_list[label_index]))

    ip_lengths_train,ip_lengths_test,payload_train,payload_test,label_train,label_test=train_test_split(ip_lengths,payload,label,train_size=train_size)
    os.makedirs(os.path.join(npy_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(npy_path, 'test'), exist_ok=True)
    np.save(os.path.join(npy_path, 'train/','ip_lengths.npy'), ip_lengths_train)
    np.save(os.path.join(npy_path, 'test/','ip_lengths.npy'), ip_lengths_test)

    np.save(os.path.join(npy_path, 'train/','payload.npy'), payload_train)
    np.save(os.path.join(npy_path, 'test/','payload.npy'), payload_test)

    np.save(os.path.join(npy_path, 'train/','label.npy'), label_train)
    np.save(os.path.join(npy_path, 'test/','label.npy'),label_test)
    logger.info("数据集划分完成")




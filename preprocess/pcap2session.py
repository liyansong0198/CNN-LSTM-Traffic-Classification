

import os
from sequence_payload.TrafficLog.setLog import logger
from flowcontainer.extractor import extract
import numpy as np

def hex_to_dec(hex_str,target_length):
    dec_list = []
    for i in range(0, len(hex_str), 2):
        dec_list.append(int(hex_str[i:i+2], 16))
    dec_list=pad_or_truncate(dec_list,target_length)
    return dec_list


def pad_or_truncate_seq(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))



# 重写，增加添加iat的功能
def get_payload_length(pcap,n,m,seq_len):
    result = extract(pcap,extension=['tcp.payload','udp.payload'])
    # 假设有k个流
    pay_load=[]
    seq_lengths=[]
    # 每一个流
    for key in result:
        value=result[key]
        packet_num=0
        if 'tcp.payload' in value.extension:
            # 提取tcp负载
            tcp_payload=[]
            for packet in value.extension['tcp.payload']:
                if packet_num<n:
                    # packet[0]是负载，1是标注该报文在流的顺序
                    load = packet[0]
                    tcp_payload.extend(hex_to_dec(load,m))
                    packet_num+=1
                else:
                    break
            # 当前包数太少，加0
            if packet_num<n:
                tcp_payload = pad_or_truncate(tcp_payload,m*n)
            pay_load.append(tcp_payload)
            ip_len = value.ip_lengths
            ip_len = pad_or_truncate_seq(ip_len,seq_len)
            seq_lengths.append(ip_len)
        elif 'udp.payload' in value.extension:
            # 提取ucp负载
            udp_payload=[]
            for packet in value.extension['udp.payload']:
                if packet_num<n:
                    # packet[0]是负载，1是标注该报文在流的顺序
                    load = packet[0]
                    udp_payload.extend(hex_to_dec(load,m))
                    packet_num+=1
                else:
                    break
            # 当前包数太少，加0
            if packet_num<n:
                udp_payload = pad_or_truncate(udp_payload,m*n)
            pay_load.append(udp_payload)
            ip_len = value.ip_lengths
            ip_len = pad_or_truncate_seq(ip_len,seq_len)
            seq_lengths.append(ip_len)
    pay_load=np.array(pay_load)
    seq_lengths=np.array(seq_lengths)
    return np.uint8(pay_load),seq_lengths


# 重写，增加添加iat的功能
def get_payload_length_timedelta(pcap,n,m,seq_len):
    result = extract(pcap,extension=['tcp.payload','udp.payload'])
    # 假设有k个流
    pay_load=[]
    seq_lengths=[]
    # 每一个流
    for key in result:
        value=result[key]
        if len(value.lengths) < 2:
            continue
        if len(value.timestamps) == 0:
            continue
        packet_num=0
        if 'tcp.payload' in value.extension:
            # 提取tcp负载
            tcp_payload=[]
            for packet in value.extension['tcp.payload']:
                if packet_num<n:
                    # packet[0]是负载，1是标注该报文在流的顺序
                    load = packet[0]
                    tcp_payload.extend(hex_to_dec(load,m))
                    packet_num+=1
                else:
                    break
            # 当前包数太少，加0
            if packet_num<n:
                tcp_payload = pad_or_truncate(tcp_payload,m*n)
            pay_load.append(tcp_payload)
            ip_len = value.ip_lengths
            ip_len = pad_or_truncate_seq(ip_len,seq_len)
            seq_lengths.append(ip_len)
        elif 'udp.payload' in value.extension:
            # 提取ucp负载
            udp_payload=[]
            for packet in value.extension['udp.payload']:
                if packet_num<n:
                    # packet[0]是负载，1是标注该报文在流的顺序
                    load = packet[0]
                    udp_payload.extend(hex_to_dec(load,m))
                    packet_num+=1
                else:
                    break
            # 当前包数太少，加0
            if packet_num<n:
                udp_payload = pad_or_truncate(udp_payload,m*n)
            pay_load.append(udp_payload)
            ip_len = value.ip_lengths
            ip_len = pad_or_truncate_seq(ip_len,seq_len)
            seq_lengths.append(ip_len)
    pay_load=np.array(pay_load)
    seq_lengths=np.array(seq_lengths)
    return np.uint8(pay_load),seq_lengths



def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))


def getPcapLength_Payload(pcap_folder,traffic_path, packet_num,byte_num,seq_length):

    for (root, _, files) in os.walk(pcap_folder):
        for Ufile in files:
            pcap_file_path = os.path.join(root, Ufile) # pcap 文件的完整路径
            pcap_name = Ufile.split('.')[0] # pcap 文件的名字
            pcap_suffix = Ufile.split('.')[1] # 文件的后缀名
            dir_suffix = root.split("\\")[-1]
            try:
                assert pcap_suffix == 'pcap'
            except:
                logger.warning('查看 pcap 文件的后缀')
            dir_name = os.path.join(traffic_path, dir_suffix,pcap_name)
            os.makedirs(dir_name, exist_ok=True) # 新建文件夹
            payload_content,sequence=get_payload_length(pcap_file_path,n=packet_num,m=byte_num,seq_len=seq_length)
            np.save(os.path.join(dir_name,'payload.npy'), payload_content)
            np.save(os.path.join(dir_name, 'ip_lengths.npy'), sequence)
            # os.remove(pcap_file_path)
            logger.info('处理完成文件 {}'.format(Ufile))
    logger.info('完成 pcap 的有效载荷与ip包长序列的提取.')
    logger.info('============\n')



def getPcapLength_Payload_puls(pcap_folder,traffic_path, packet_num,byte_num,seq_length):
    for (root, _, files) in os.walk(pcap_folder):
        for Ufile in files:
            pcap_file_path = os.path.join(root, Ufile) # pcap 文件的完整路径
            pcap_name = Ufile.split('.')[0] # pcap 文件的名字
            pcap_suffix = Ufile.split('.')[1] # 文件的后缀名
            dir_suffix = root.split("\\")[-1]
            try:
                assert pcap_suffix == 'pcap'
            except:
                logger.warning('查看 pcap 文件的后缀')
            dir_name = os.path.join(traffic_path, dir_suffix,pcap_name)
            os.makedirs(dir_name, exist_ok=True) # 新建文件夹
            payload_content,sequence=get_payload_length(pcap_file_path,n=packet_num,m=byte_num,seq_len=seq_length)
            np.save(os.path.join(dir_name,'payload.npy'), payload_content)
            np.save(os.path.join(dir_name, 'ip_lengths.npy'), sequence)
            # os.remove(pcap_file_path)
            logger.info('处理完成文件 {}'.format(Ufile))
    logger.info('完成 pcap 的有效载荷与ip包长序列的提取.')
    logger.info('============\n')

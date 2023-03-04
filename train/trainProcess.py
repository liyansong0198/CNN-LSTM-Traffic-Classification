'''
@Author: WANG Maonan
@Date: 2021-01-07 17:03:15
@Description: 模型训练的流程, 这里是一个 epoch 的训练流程
@LastEditTime: 2021-02-06 17:00:43
'''

from sequence_payload.utils.helper import AverageMeter, accuracy
from sequence_payload.TrafficLog.setLog import logger


def train_process(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    """训练一个 epoch 的流程

    Args:
        train_loader (dataloader): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        epoch (int): 当前所在的 epoch
        device (torch.device): 是否使用 gpu
        print_freq ([type]): [description]
    """
    losses = AverageMeter()  # 在一个 train loader 中的 loss 变化
    top1 = AverageMeter()  # 记录在一个 train loader 中的 accuracy 变化

    model.train()  # 切换为训练模型

    for i, (pcap, seq,target) in enumerate(train_loader):
        pcap = pcap.reshape(-1,1,1024)
        seq = seq.reshape(-1,64,1)
        pcap = pcap.to(device)
        seq = seq.to(device)
        target = target.to(device)

        output = model(pcap,seq)  # 得到模型预测结果
        loss = criterion(output, target)  # 计算 loss

        # 计算准确率, 记录 loss 和 accuracy
        # print(pcap.size(0))
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), pcap.size(0))
        top1.update(prec1[0].item(), pcap.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1))
    return losses.val,top1.val

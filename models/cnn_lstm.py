"""
cnn处理负载
lstm处理包长序列
"""

import torch
import torch.nn as nn


class Cnn_Lstm(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers,bidirectional,num_classes=12):
        super(Cnn_Lstm, self).__init__()
        # rnn配置
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=bidirectional,batch_first=True)
        self.fc0 = nn.Linear(hidden_size, num_classes)
        self.fc1= nn.Linear(hidden_size*2,num_classes)

        self.cnn_feature = nn.Sequential(
            # 卷积层1
            nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12),  # (1,1024)->(32,1024)
            nn.BatchNorm1d(32),  # 加上BN的结果
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (32,1024)->(32,342)

            # 卷积层2
            nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12),  # (32,342)->(64,342)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (64,342)->(64,114)
        )
        # 全连接层
        self.cnn_classifier = nn.Sequential(
            # 64*114
            nn.Flatten(),
            nn.Linear(in_features=64*114, out_features=1024), # 784:88*64, 1024:114*64, 4096:456*64
        )

        self.cnn=nn.Sequential(
            self.cnn_feature,
            self.cnn_classifier,
        )


        self.rnn = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True),
        )
        self.classifier=nn.Sequential(
            nn.Linear(in_features=2048,out_features=num_classes),
            # nn.Dropout(p=0.7),
            # nn.Linear(in_features=1024,out_features=num_classes)
        )


    def forward(self, x_payload,x_sequence):
        x_payload=self.cnn(x_payload)
        x_sequence=self.rnn(x_sequence)
        x_sequence=x_sequence[0][:, -1, :]
        x=torch.cat((x_payload,x_sequence),1)
        x=self.classifier(x)
        return x


def cnn_rnn(model_path, pretrained=False, **kwargs):
    """
    CNN 1D model architecture

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Cnn_Lstm(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model


# 仅仅是CNN
class Cnn(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers,bidirectional,num_classes=12):
        super(Cnn, self).__init__()
        # rnn配置
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=bidirectional,batch_first=True)
        self.fc0 = nn.Linear(hidden_size, num_classes)
        self.fc1= nn.Linear(hidden_size*2,num_classes)

        self.cnn_feature = nn.Sequential(
            # 卷积层1
            nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12),  # (1,1024)->(32,1024)
            nn.BatchNorm1d(32),  # 加上BN的结果
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (32,1024)->(32,342)

            # 卷积层2
            nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12),  # (32,342)->(64,342)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (64,342)->(64,114)
        )
        # 全连接层
        self.cnn_classifier = nn.Sequential(
            # 64*114
            nn.Flatten(),
            nn.Linear(in_features=64*114, out_features=1024), # 784:88*64, 1024:114*64, 4096:456*64
        )

        self.cnn=nn.Sequential(
            self.cnn_feature,
            self.cnn_classifier,
        )
        self.classifier=nn.Sequential(
            nn.Linear(in_features=1024,out_features=num_classes),
            # nn.Dropout(p=0.7),
            # nn.Linear(in_features=1024,out_features=num_classes)
        )


    def forward(self, x_payload,x_sequence):
        x_payload=self.cnn(x_payload)
        x=self.classifier(x_payload)
        return x_payload


def cnn(model_path, pretrained=False, **kwargs):
    model = Cnn(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model


class Lstm(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers,bidirectional,num_classes=12):
        super(Lstm, self).__init__()
        # rnn配置
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,bidirectional=bidirectional,batch_first=True)
        self.fc0 = nn.Linear(hidden_size, num_classes)
        self.fc1= nn.Linear(hidden_size*2,num_classes)

        self.rnn = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True),
        )
        self.classifier=nn.Sequential(
            nn.Linear(in_features=1024,out_features=num_classes),
            # nn.Dropout(p=0.7),
            # nn.Linear(in_features=1024,out_features=num_classes)
        )


    def forward(self, x_payload,x_sequence):
        x_sequence=self.rnn(x_sequence)
        x_sequence=x_sequence[0][:, -1, :]
        x=self.classifier(x_sequence)
        return x


def rnn(model_path, pretrained=False, **kwargs):
    """
    CNN 1D model architecture

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Lstm(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model


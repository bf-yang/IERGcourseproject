import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
loss_f = nn.CrossEntropyLoss()


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = torch.nn.Linear(768, 128)  # hidden layer
        # self.hidden2 = torch.nn.Linear(128, 64)   # hidden layer
        self.out = torch.nn.Linear(128, 12)       # hidden layer


    def forward(self, x):
        x = F.relu(self.hidden1(x))
        # x = F.relu(self.hidden2(x))    
        x = self.out(x)
        return x
    




class CNN(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 256, (6,1), (3,1), (1,0))
        self.layer4 = nn.Linear(46080, 768)
        self.fc = nn.Linear(768, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        out = self.fc(x)
        # print(x.shape)

        return out, x
    


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, (3,1), 1, (1,0)),
            nn.BatchNorm2d(output_channel),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
        )
    def forward(self, x):
        identity = self.shortcut(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = x + identity
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResNet, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 256, (6,1), (3,1), (1,0))
        self.fc = nn.Linear(46080, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        # print(x.size())

        out = self.fc(x)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda()

        return out, x
    
    
    
class ResNet1(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResNet1, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 256, (6,1), (3,1), (1,0))
        self.fc1 = nn.Linear(46080, 768)
        self.fc = nn.Linear(768, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc1(x)
        out = self.fc(x)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda()

        return out, x
    



    
class CNN_clip(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN_clip, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 256, (6,1), (3,1), (1,0))
        self.layer4 = nn.Linear(46080, 768)
        self.fc = nn.Linear(768, num_classes)
        
        self.logit_scale_im = nn.Parameter(torch.ones([]) * 0.1)
        self.logit_scale_tm = nn.Parameter(torch.ones([]) * 0.1)

        
    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, image_features:Optional[torch.Tensor] = None,
                text_features:Optional[torch.Tensor] = None,
                mode:Optional[str] = None):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        imu_features = self.layer4(x)
        out = self.fc(imu_features)
        
        if image_features is None or text_features is None:
            return out, imu_features # text stage

        # Train stage
        # Cosine similarity as logits
        logit_scale_im = self.logit_scale_im.exp()
        logit_scale_tm = self.logit_scale_tm.exp()
        
        # image -> imu
        logits_per_image_1 = logit_scale_im * image_features @ imu_features.t()
        logits_per_image_2 = logit_scale_im * imu_features @ image_features.t()
        
        # text -> imu
        logits_per_text_1 = logit_scale_tm * text_features @ imu_features.t() # [256,256]
        logits_per_text_2 = logit_scale_tm * imu_features @ text_features.t() # [256,256]

        batch_size = imu_features.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device) # [256]

        loss_image_imu_1 = loss_f(logits_per_image_1, ground_truth) # tensor, (20.0042)
        loss_image_imu_2 = loss_f(logits_per_image_2, ground_truth)

        loss_text_imu_1 = loss_f(logits_per_text_1, ground_truth)
        loss_text_imu_2 = loss_f(logits_per_text_2, ground_truth)

        loss_mse = nn.MSELoss()(imu_features, image_features)
     
     
        if mode == 'text':
            return out, imu_features, (loss_text_imu_1+loss_text_imu_2)/2
        if mode == 'image':
            return out, imu_features, (loss_image_imu_1+loss_image_imu_2)/2
        if mode == 'text+image':
            return out, imu_features, (loss_text_imu_1+loss_text_imu_2)/2, loss_mse
    
    
    
    
class ResNet_clip(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(ResNet_clip, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 256, (6,1), (3,1), (1,0))
        self.layer4 = nn.Linear(46080, 768)
        self.fc = nn.Linear(768, num_classes)

        self.logit_scale_im = nn.Parameter(torch.ones([]) * 0.1)
        self.logit_scale_tm = nn.Parameter(torch.ones([]) * 0.1)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x, image_features:Optional[torch.Tensor] = None,
                text_features:Optional[torch.Tensor] = None,
                mode:Optional[str] = None):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        # print(x.size())

        imu_features = self.layer4(x)
        out = self.fc(imu_features)
        
        if image_features is None or text_features is None:
            return out, imu_features # text stage

        # Train stage
        # Cosine similarity as logits
        logit_scale_im = self.logit_scale_im.exp()
        logit_scale_tm = self.logit_scale_tm.exp()
        
        # image -> imu
        logits_per_image_1 = logit_scale_im * image_features @ imu_features.t()
        logits_per_image_2 = logit_scale_im * imu_features @ image_features.t()
        
        # text -> imu
        logits_per_text_1 = logit_scale_tm * text_features @ imu_features.t()
        logits_per_text_2 = logit_scale_tm * imu_features @ text_features.t()

        batch_size = imu_features.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

        loss_image_imu_1 = loss_f(logits_per_image_1, ground_truth)
        loss_image_imu_2 = loss_f(logits_per_image_2, ground_truth)

        loss_text_imu_1 = loss_f(logits_per_text_1, ground_truth)
        loss_text_imu_2 = loss_f(logits_per_text_2, ground_truth)

        loss_mse = nn.MSELoss()(imu_features, image_features)


     
        if mode == 'text':
            return out, imu_features, (loss_text_imu_1+loss_text_imu_2)/2
        if mode == 'image':
            return out, imu_features, (loss_image_imu_1+loss_image_imu_2)/2
        if mode == 'text+image':
            return out, imu_features, (loss_text_imu_1+loss_text_imu_2)/2, loss_mse




class CNN_clip_zs(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN_clip_zs, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6,1), (3,1), (1,0))
        self.layer2 = self._make_layers(64, 128, (6,1), (3,1), (1,0))
        self.layer3 = self._make_layers(128, 256, (6,1), (3,1), (1,0))
        self.layer4 = nn.Linear(46080, 768)
        # self.fc = nn.Linear(768, num_classes)
        
        self.logit_scale_im = nn.Parameter(torch.ones([]) * 0.1)
        self.logit_scale_tm = nn.Parameter(torch.ones([]) * 0.1)

        
    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, image_features:Optional[torch.Tensor] = None,
                text_features:Optional[torch.Tensor] = None,
                mode:Optional[str] = None):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('aa',x.shape)
        # x = F.max_pool2d(x, (4,3))
        x = x.view(x.size(0), -1)
        imu_features = self.layer4(x)
        # out = self.fc(imu_features)
        
        if image_features is None or text_features is None:
            # return out, imu_features # text stage
            return imu_features # text stage


        # Train stage
        # Cosine similarity as logits
        logit_scale_im = self.logit_scale_im.exp()
        logit_scale_tm = self.logit_scale_tm.exp()
        
        # image -> imu
        logits_per_image_1 = logit_scale_im * image_features @ imu_features.t()
        logits_per_image_2 = logit_scale_im * imu_features @ image_features.t()
        
        # text -> imu
        logits_per_text_1 = logit_scale_tm * text_features @ imu_features.t()
        logits_per_text_2 = logit_scale_tm * imu_features @ text_features.t()

        batch_size = imu_features.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

        loss_image_imu_1 = loss_f(logits_per_image_1, ground_truth)
        loss_image_imu_2 = loss_f(logits_per_image_2, ground_truth)

        loss_text_imu_1 = loss_f(logits_per_text_1, ground_truth)
        loss_text_imu_2 = loss_f(logits_per_text_2, ground_truth)

        if mode == 'text':
            return imu_features, (loss_image_imu_1+loss_image_imu_2)/2
        if mode == 'image':
            return imu_features, (loss_text_imu_1+loss_text_imu_2)/2


    

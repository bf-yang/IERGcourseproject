# Only use IMU and ground-truth label for supervised learning
import torch
import torch.nn as nn
import torch.utils.data as Data
import os
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import argparse
import random
import logging
from model import CNN, ResNet, ResNet1
from util import AverageMeter, accuracy, adjust_learning_rate
import util

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=512, help='Batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--epoch', type=int, default=200, help='Epoch')
args = parser.parse_args()

# Params
json_path = './params.json'
params = util.Params(json_path)


# Load data
train_x = torch.from_numpy(np.load('./Datasets/x_train.npy')).float()
train_y = torch.from_numpy(np.load('./Datasets/y_train.npy')).long()
test_x = torch.from_numpy(np.load('./Datasets/x_test.npy')).float()
test_y = torch.from_numpy(np.load('./Datasets/y_test.npy')).long()

train_x = torch.unsqueeze(train_x, 1)
test_x = torch.unsqueeze(test_x, 1)
num_classes = len(Counter(train_y.tolist()))


train_dataset = Data.TensorDataset(train_x, train_y)
test_dataset = Data.TensorDataset(test_x, test_y)

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True)


# Model definition
# model = CNN(input_channel=1, num_classes=num_classes).to(device)
model_dir = './experiments_imuonly_new'
model = ResNet1(input_channel=1, num_classes=num_classes).to(device)
# print(model)

loss_ce = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


best_val_acc = 0.0
for epoch in range(args.epoch):
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    scheduler = StepLR(optimizer, step_size=100, gamma=0.2) 
    
    # set model to training mode
    model.train()
    for i, (train_batch, labels_batch) in enumerate(train_loader):
        train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
        # print(train_batch.shape,labels_batch.shape)
        output_batch, _ = model(train_batch)  
        # print(output_batch.shape,labels_batch.shape)

        loss = loss_ce(output_batch, labels_batch)
                
        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()
        
        # Compute acc
        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))  # output:[32,100], target:[32]
        top1.update(acc1[0], train_batch.size(0))
        top5.update(acc5[0], train_batch.size(0)) 
        
        # evaluate
        if i % params.save_summary_steps == 0:
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            print( 'Epoch [{}/{}], Iteration [{}/{}]'.format(epoch+1, params.num_epochs+1, i + 1, len(train_loader)),' Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5), 'Loss {}'.format(loss))


    
    # Test evaluate
    top1_eval = AverageMeter()
    top5_eval = AverageMeter()
    model.eval()
    for i, (val_batch, labels_batch) in enumerate(test_loader):
        val_batch, labels_batch = val_batch.cuda(), labels_batch.cuda()
                
        output_batch, _  = model(val_batch)
    
        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))  # output:[32,100], target:[32]
        top1_eval.update(acc1[0], val_batch.size(0))
        top5_eval.update(acc5[0], val_batch.size(0)) 
    print(' * Acc@1 {top1_eval.avg:.3f} Acc@5 {top5_eval.avg:.3f}'.format(top1_eval=top1_eval, top5_eval=top5_eval))


    # evaluate
    if i % params.save_summary_steps == 0:
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        print( 'Epoch [{}/{}], Iteration [{}/{}]'.format(epoch+1, params.num_epochs+1, i + 1, len(test_loader)),' Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5), 'Loss {}'.format(loss))


    # Save model & metrics    
    val_acc = float(top1_eval.avg.cpu().numpy())
    is_best = val_acc>=best_val_acc
    
    # Save weights
    util.save_checkpoint({'state_dict': model.state_dict()},is_best=is_best,checkpoint=model_dir)
    
    # Save val metrics
    dict_save = {'Acc@1':top1_eval.avg.cpu().numpy(), 'Acc@5':top5_eval.avg.cpu().numpy()}
    # If best_eval, best_save_path
    if is_best:
        logging.info("- Found new best accuracy")
        best_val_acc = val_acc

        # Save best val metrics in a json file in the model directory
        best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
        util.save_dict_to_json(dict_save, best_json_path)

    # Save latest val metrics in a json file in the model directory
    last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
    util.save_dict_to_json(dict_save, last_json_path)

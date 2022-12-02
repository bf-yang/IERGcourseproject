# IMU + IMAGE + TEXT contrastive learning
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
from model import CNN, ResNet, CNN_clip, ResNet_clip
from util import AverageMeter, accuracy, adjust_learning_rate
import util
import clip
import torch
from PIL import Image

# Random seed
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
parser.add_argument('--bs', type=int, default=256, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--epoch', type=int, default=200, help='Epoch')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha')

args = parser.parse_args()
print(args.bs,args.lr,args.epoch,args.alpha)

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
# len_train, len_test = len(train_y),  len(test_y)



classnames = ['lying', 'sitting','standing','walking', 'running','cycling','Nordic_walking','ascending_stairs','descending_stairs',
              'vacuum_cleaning','ironing','rope_jumping']
# templates = ['The human is {}', 'This is a human action of {}', 'Human is {}',
#              'An activity data of {}', 'An action data of {}', 
#              'An human activity data of {}', 'An human action data of {}', 
#              'An IMU data of {}, a type of human activity',
#              'A type of {} activity', 'A type of {} action'
#              ]
# text, labels = [], []
# for action_label, classname in enumerate(classnames):
#     texts_class = [template.format(classname) for template in templates]
#     for prompt in templates:
#         labels.append(action_label)
#     text = text + texts_class
    
activityID = ['lying', 'sitting','standing','walking', 'running','cycling','Nordic walking','ascending stairs','descending stairs',
              'vacuum cleaning','ironing','rope jumping']
prompt = 'The human is '
text = []
for word in activityID:
    text.append(prompt + word)
    # text.append(word)

# Load CLIP model
model, preprocess = clip.load("ViT-L/14", device=device)

# Text features of CLIP
text_tokenize = clip.tokenize(text).to(device)
text_features = model.encode_text(text_tokenize)
text_features = text_features.detach().float()


# # Text features ensumble
# N_ensem = len(templates)
# text_features_ensem = torch.zeros((len(classnames), text_features.shape[1])).to(device)
# for i in range(len(classnames)):
#     start = N_ensem * i
#     end = start + N_ensem
#     text_features_ensem[i,:] = text_features[start:end,:].mean(0)
# text_features = text_features_ensem
    
# Prepare action images
image_features = []
for action in classnames:
    if action=='Nordic_walking' or action=='rope_jumping' or action=='vacuum_cleaning':
        filename = './action_images/' + action + '.jpg'
    else:
        filename = './action_images/' + action + '.png'
    # load image
    image = preprocess(Image.open(filename)).unsqueeze(0).to(device) 
    features = model.encode_image(image.to(device)).cpu().detach().numpy() 
    image_features.append(features[0])
image_features = torch.Tensor(image_features).to(device)


# Assign image features & text features to each sample
train_img_features = torch.zeros(len(train_y), image_features.shape[1])
train_txt_features = torch.zeros(len(train_y), text_features.shape[1])

for idx in range(len(train_y)):
    action_label = train_y[idx]
    train_img_features[idx,:] = image_features[action_label, :]

for idx in range(len(train_y)):
    action_label = train_y[idx]
    train_txt_features[idx,:] = text_features[action_label, :]



train_dataset = Data.TensorDataset(train_x, train_y, train_img_features, train_txt_features)
test_dataset = Data.TensorDataset(test_x, test_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True)


# alpha = 0.7  # ratio of text/image
alpha = args.alpha
# Model definition
model = CNN_clip(input_channel=1, num_classes=num_classes).to(device)
model = ResNet_clip(input_channel=1, num_classes=num_classes).to(device)

model_dir = './experiments_imutextimg'+str(alpha)
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
    for i, (x, y, f_img, f_txt) in enumerate(train_loader):
        x, y, f_img, f_txt = x.cuda(), y.cuda(), f_img.cuda(), f_txt.cuda()
        output, imu_features, loss_contrastive, loss_mse = model(x, f_img, f_txt, 'text+image')
        
        score = imu_features @ text_features.t()
        similarity = (100.0 * imu_features @ text_features.t()).softmax(dim=-1)
        pred_mt = torch.max(similarity, 1)[1].cpu().numpy()
        
        
        loss = alpha*loss_contrastive + (1-alpha)*loss_mse
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        # Compute acc
        acc1, acc5 = accuracy(similarity, y, topk=(1, 5))  # output:[32,100], target:[32]
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0)) 
        
 
        # evaluate
        if i % params.save_summary_steps == 0:
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            print( 'Epoch [{}/{}], Iteration [{}/{}]'.format(epoch+1, params.num_epochs+1, i + 1, len(train_loader)),' Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5), 'Loss_contra {}'.format(loss_contrastive), 'Loss_mse {}'.format(loss_mse))

    
    
    # Test evaluate
    top1_eval = AverageMeter()
    top5_eval = AverageMeter()
    model.eval()
    for i, (val_batch, labels_batch) in enumerate(test_loader):
        val_batch, labels_batch = val_batch.cuda(), labels_batch.cuda()
                
        output_batch, imu_features = model(val_batch)
        
        score = imu_features @ text_features.t()
        similarity = (100.0 * imu_features @ text_features.t()).softmax(dim=-1)
        
    
        acc1, acc5 = accuracy(similarity, labels_batch, topk=(1, 5))  # output:[32,100], target:[32]
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

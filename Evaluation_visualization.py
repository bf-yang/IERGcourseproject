import matplotlib.pyplot as plt
from matplotlib import cm
import json
import os
import shutil
import torch
from sklearn.metrics import f1_score
import numpy as np

def adjust_learning_rate(optimizer, epoch, lr_init):
    lr = lr_init * (0.1 ** (epoch // 50))
    optimizer.param_groups[0]['lr'] = lr
    
    
def plot_with_labels(lowDWeights, labels):
        plt.cla()
        X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
        for x, y, s in zip(X, Y, labels):
            c = cm.rainbow(int(255 * s / 11))
            # plt.scatter(x, y, color=c)
            plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); 
        plt.title('Visualize features'); plt.show(); plt.pause(0.01)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
        
        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def eval_metrics(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        # F1 score
        output_new = np.argmax(output.cpu(), axis=1)
        macrof1 = f1_score(target.cpu(), output_new, average='macro')
        microf1 = f1_score(target.cpu(), output_new, average='micro')


        return res[0], res[1], macrof1, microf1


import matplotlib.pyplot as plt

def draw_scatter(X_tsne, annotation_val, title):
    plt.figure()
    plt.tick_params(labelsize=10)
    axes = plt.subplot(111)
    type1_x, type1_y = [], []
    type2_x, type2_y = [],[]
    type3_x, type3_y = [],[]
    type4_x, type4_y = [],[]
    type5_x, type5_y = [],[]
    type6_x, type6_y = [],[]
    type7_x, type7_y = [],[]
    type8_x, type8_y = [],[]
    type9_x, type9_y = [],[]
    type10_x, type10_y = [],[]
    type11_x, type11_y = [],[]
    type12_x, type12_y = [],[]

    for i in range(len(annotation_val)):
        if annotation_val[i] == 0:
            type1_x.append(X_tsne[i][0])
            type1_y.append(X_tsne[i][1])

        if annotation_val[i] == 1:
            type2_x.append(X_tsne[i][0])
            type2_y.append(X_tsne[i][1])

        if annotation_val[i] == 2:
            type3_x.append(X_tsne[i][0])
            type3_y.append(X_tsne[i][1])

        if annotation_val[i] == 3:
            type4_x.append(X_tsne[i][0])
            type4_y.append(X_tsne[i][1])

        if annotation_val[i] == 4:
            type5_x.append(X_tsne[i][0])
            type5_y.append(X_tsne[i][1])
            
        if annotation_val[i] == 5:
            type6_x.append(X_tsne[i][0])
            type6_y.append(X_tsne[i][1])

        if annotation_val[i] == 6:
            type7_x.append(X_tsne[i][0])
            type7_y.append(X_tsne[i][1])

        if annotation_val[i] == 7:
            type8_x.append(X_tsne[i][0])
            type8_y.append(X_tsne[i][1])

        if annotation_val[i] == 8:
            type9_x.append(X_tsne[i][0])
            type9_y.append(X_tsne[i][1])

        if annotation_val[i] == 9:
            type10_x.append(X_tsne[i][0])
            type10_y.append(X_tsne[i][1])
            
        if annotation_val[i] == 10:
            type11_x.append(X_tsne[i][0])
            type11_y.append(X_tsne[i][1])

        if annotation_val[i] == 11:
            type12_x.append(X_tsne[i][0])
            type12_y.append(X_tsne[i][1])
    size = 10
    type1 = axes.scatter(type1_x, type1_y, s=size)
    type2 = axes.scatter(type2_x, type2_y, s=size)
    type3 = axes.scatter(type3_x, type3_y, s=size)
    type4 = axes.scatter(type4_x, type4_y, s=size)
    type5 = axes.scatter(type5_x, type5_y, s=size)
    type6 = axes.scatter(type6_x, type6_y, s=size)
    type7 = axes.scatter(type7_x, type7_y, s=size)
    type8 = axes.scatter(type8_x, type8_y, s=size)
    type9 = axes.scatter(type9_x, type9_y, s=size)
    type10 = axes.scatter(type10_x, type10_y, s=size)
    type11 = axes.scatter(type11_x, type11_y, s=size)
    type12 = axes.scatter(type12_x, type12_y, s=size)   
    # axes.legend((type1, type2, type3, type4, type5, type6, type7, type8, type9, type10, type11, type12), ('lying', 'sitting','standing','walking', 'running','cycling','Nordic_walking','ascending_stairs','descending_stairs',
    #           'vacuum_cleaning','ironing','rope_jumping'), prop={'size': 8}, loc='lower right')
    # plt.title(title)
    plt.xlabel("Component 1", fontsize=16)
    plt.ylabel("Component 2", fontsize=16)
    plt.show()


    
class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

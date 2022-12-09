import os
import sys
import datetime
import math
import dataset
import argparse
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler
from apex import amp
from model.CamoFormer import CamoFormer 
import matplotlib.pyplot as plt
plt.ion()

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '1'



def loss_cal(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def train(Dataset, parser):
    
    args   = parser.parse_args()
    _MODEL_ = args.model
    _DATASET_ = args.dataset
    _TESTDATASET_ = args.test_dataset
    _LR_ = args.lr
    _DECAY_ = args.decay
    _MOMEN_ = args.momen
    _BATCHSIZE_ = args.batchsize
    _EPOCH_ = args.epoch
    _SAVEPATH_ = args.savepath
    _VALID_ = args.valid 
    _WEIGHT_ = args.weight
    _PRETRAINPATH_ = args.pretrain_path

    print(args)

    cfg    = Dataset.Config(datapath=_DATASET_, savepath=_SAVEPATH_, mode='train', batch=_BATCHSIZE_, lr=_LR_, momen=_MOMEN_, decay=_DECAY_, epoch=_EPOCH_)

    data   = Dataset.Data(cfg, _MODEL_)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=6)
    ## network
    net = CamoFormer(cfg, _PRETRAINPATH_)
    net = net.cuda()

    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    
    for name, param in net.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            pass
        elif 'encoder' in name:
            base.append(param)
        elif 'network' in name:
            base.append(param)     
        else:
            head.append(param)

    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O1') 
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = 0.5*(1 + math.cos(math.pi * (epoch ) / (cfg.epoch)))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = 0.5*(1 + math.cos(math.pi * (epoch ) / (cfg.epoch)))*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda(), mask.cuda()   
            P5, P4, P3, P2, P1 = net(image)

            loss5  = loss_cal(P5, mask)+F.binary_cross_entropy_with_logits(P5,mask)
            loss4  = loss_cal(P4, mask)+F.binary_cross_entropy_with_logits(P4,mask)
            loss3  = loss_cal(P3, mask)+F.binary_cross_entropy_with_logits(P3,mask)
            loss2  = loss_cal(P2, mask)+F.binary_cross_entropy_with_logits(P2,mask)
            loss1  = loss_cal(P1, mask)+F.binary_cross_entropy_with_logits(P1,mask)
            
            loss = _WEIGHT_[0]*loss5 + _WEIGHT_[1]*loss4 + _WEIGHT_[2]*loss3 + _WEIGHT_[3]*loss2 + _WEIGHT_[4]*loss1  
            
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            global_step += 1
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss1=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f |'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item()))


        if epoch % 5 == 0 or epoch > 55:
            torch.save(net.state_dict(), cfg.savepath+'/'+_MODEL_+str(epoch+1))
            # 'CHAMELEON','COD10K','NC4K','CAMO'
            for path in ['COD10K']: 
                path=_TESTDATASET_+'/'+path
                t = Valid(dataset, path, epoch, 'CamoFormer','output/checkpoint/CamoFormer/CamoFormer/' )
                t.save()
            os.system('bash evaltools/valid_eval.sh '+'output/Prediction/CamoFormer-epoch'+str(epoch+1)+' '+_TESTDATASET_+' CamoFormer'+str(epoch+1))
            
def test(dataset,parser):
    args   = parser.parse_args()
    _TESTDATASET_ = args.test_dataset
    _CKPT_ = args.ckpt
    

    for path in ['CHAMELEON','COD10K','NC4K','CAMO']:
        path=_TESTDATASET_+'/'+path
        t = Valid(dataset, path, 0, 'CamoFormer', _CKPT_, mode='test')
        t.save()


class Valid(object):
    def __init__(self, Dataset, Path, epoch, model_name, checkpoint_path, mode='Valid'):
        ## dataset
        if mode == 'test':
            self.cfg = Dataset.Config(datapath=Path, snapshot=checkpoint_path, mode='test')
        else:
            self.cfg = Dataset.Config(datapath=Path, snapshot=checkpoint_path+model_name+str(epoch+1), mode='test')
        self.mode = mode
        self.data   = Dataset.Data(self.cfg, model_name)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = CamoFormer(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.epoch = epoch
        
    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                P5, P4, P3, P2, P1 = self.net(image, shape)
                pred = torch.sigmoid(P1[0,0]).cpu().numpy()*255
                if self.mode == 'test':
                    head = 'output/Prediction/CamoFormer-test'+'/'+ self.cfg.datapath.split('/')[-1]
                else:
                    head = 'output/Prediction/CamoFormer-epoch'+str(self.epoch+1)+'/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0].replace('.jpg','')+'.png', np.round(pred))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='CamoFormer')
    parser.add_argument("--dataset", default='dataset/TrainDataset')
    parser.add_argument("--test_dataset", default='dataset/TestDataset')
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momen", type=float, default=0.9) 
    parser.add_argument("--decay", type=float, default=1e-4)  
    parser.add_argument("--batchsize", type=int, default=14) 
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--savepath", default='output/checkpoint/CamoFormer/CamoFormer')
    parser.add_argument("--weight", default=[0.5,0.5,0.8,1.0,2.0])
    parser.add_argument("--valid", default=True)
    parser.add_argument("--mode", default='train')
    parser.add_argument("--ckpt", default='CamoFormer.pth')
    parser.add_argument("--pretrain_path",default=None)
    args   = parser.parse_args()
    if args.mode == 'train':
        train(dataset, parser)
    else:
        test(dataset, parser)
       
import os
import sys
import cv2
from tqdm import tqdm
import metrics 
import json
import argparse
import numpy as np

def Borders_Capture(gt,pred,dksize=15):
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img=gt.copy()
    img[:]=0
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    kernel = np.ones((dksize, dksize), np.uint8)
    img_dilate = cv2.dilate(img, kernel)

    res = cv2.bitwise_and(img_dilate, gt)
    b, g, r = cv2.split(res)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    merge = cv2.merge((b, g, r, alpha))

    resp = cv2.bitwise_and(img_dilate, pred)
    b, g, r = cv2.split(resp)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    mergep = cv2.merge((b, g, r, alpha))

    merge = cv2.cvtColor(merge, cv2.COLOR_RGB2GRAY)
    mergep = cv2.cvtColor(mergep, cv2.COLOR_RGB2GRAY)
    return merge,mergep,np.sum(img_dilate)/255

def eval(parser, dataset):
    args = parser.parse_args()

    FM = metrics.Fmeasure_and_FNR()
    WFM = metrics.WeightedFmeasure()
    SM = metrics.Smeasure()
    EM = metrics.Emeasure()
    MAE = metrics.MAE()

    BR_MAE = metrics.MAE()
    BR_wF = metrics.WeightedFmeasure()

    model = args.model
    gt_root = args.GT_root
    pred_root = args.pred_root

    gt_root = os.path.join(gt_root, dataset)
    gt_root = os.path.join(gt_root, 'GT')
    pred_root = os.path.join(pred_root, dataset)

    gt_name_list = sorted(os.listdir(pred_root))

    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_width, gt_height = gt.shape
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_width, pred_height = pred.shape
        if gt.shape != pred.shape:
            cv2.imwrite( os.path.join(pred_root, gt_name), cv2.resize(pred, gt.shape[::-1]))
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

        if args.BR == 'on':
            BR_gt, BR_pred, area=Borders_Capture(cv2.imread(gt_path), cv2.imread(pred_path), int(args.br_rate))
            BR_MAE.step(pred=BR_pred, gt=BR_gt,area=area)
            BR_wF.step(pred=BR_pred, gt=BR_gt)

    fm = FM.get_results()[0]['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    fnr = FM.get_results()[1]
    model_r = str(args.model)
    Smeasure_r = str(sm.round(3))
    Wmeasure_r = str(wfm.round(3))
    MAE_r = str(mae.round(3))
    adpEm_r = str(em['adp'].round(3))
    meanEm_r = str('-' if em['curve'] is None else em['curve'].mean().round(3))
    maxEm_r = str('-' if em['curve'] is None else em['curve'].max().round(3))
    adpFm_r = str(fm['adp'].round(3))  
    meanFm_r = str(fm['curve'].mean().round(3))
    maxFm_r = str(fm['curve'].max().round(3))
    fnr_r = str(fnr.round(3))

    if args.BR == 'on':
        BRmae= BR_MAE.get_results()['mae']
        BRmae_r = str(BRmae.round(3))
        BRwF = BR_wF.get_results()['wfm']
        BRwF_r = str(BRwF.round(3))
        eval_record = str(        
            'Model:'+ model_r + ','+
            'Dataset:'+ dataset + '||'+
            'Smeasure:'+ Smeasure_r + '; '+
            'meanEm:'+ meanEm_r + '; '+
            'wFmeasure:'+ Wmeasure_r + '; '+
            'MAE:'+ MAE_r + '; '+
            'fnr:'+ fnr_r + ';' +
            'adpEm:'+ adpEm_r + '; '+
            'meanEm:'+ meanEm_r + '; '+
            'maxEm:'+ maxEm_r + '; '+
            'adpFm:'+ adpFm_r + '; '+
            'meanFm:'+ meanFm_r + '; '+
            'maxFm:'+ maxFm_r+ ';' +
            'BR'+str(args.br_rate)+'_mae:' + BRmae_r + ';' +
            'BR'+str(args.br_rate)+'_wF:' + BRwF_r
            )
    else:
        eval_record = str(        
            'Model:'+ model_r + ','+
            'Dataset:'+ dataset + '||'+
            'Smeasure:'+ Smeasure_r + '; '+
            'meanEm:'+ meanEm_r + '; '+
            'wFmeasure:'+ Wmeasure_r + '; '+
            'MAE:'+ MAE_r + '; '+
            'fnr:'+ fnr_r + ';' +
            'adpEm:'+ adpEm_r + '; '+
            'meanEm:'+ meanEm_r + '; '+
            'maxEm:'+ maxEm_r + '; '+
            'adpFm:'+ adpFm_r + '; '+
            'meanFm:'+ meanFm_r + '; '+
            'maxFm:'+ maxFm_r
            )


    print(eval_record)
    print('#'*50)
    if args.record_path is not None:
        txt = args.record_path
    else:
        txt = 'output/eval_record.txt'
    f = open(txt, 'a')
    f.write(eval_record)
    f.write("\n")
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='CamoFormer')
    parser.add_argument("--pred_root", default='Prediction/CamoFormer')
    parser.add_argument("--GT_root", default='Dataset/TestData')
    parser.add_argument("--record_path", default=None)
    parser.add_argument("--BR", default='off')
    parser.add_argument("--br_rate", default=15)
    args = parser.parse_args()
    datasets = ['NC4K', 'COD10K', 'CAMO', 'CHAMELEON']
    existed_pred = os.listdir(args.pred_root)
    for dataset in datasets:  
        if dataset in existed_pred:
            eval(parser, dataset)

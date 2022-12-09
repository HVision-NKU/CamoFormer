python main.py \
    --model 'CamoFormer' \
    --dataset 'dataset/TrainDataset' \
    --test_dataset 'dataset/TestDataset/' \
    --pretrain_path 'checkpoint/pvt_v2_b4.pth' \
    --lr 1e-2 \
    --decay 2e-4 \
    --momen 0.9 \
    --batchsize 8 \
    --savepath 'output/checkpoint/CamoFormer/CamoFormer/' \


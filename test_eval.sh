python main.py \
    --ckpt 'checkpoint/CamoFormer56'\
    --mode 'test'

python evaltools/eval.py   \
    --GT_root  'dataset/TestDataset' \
    --pred_root 'output/Prediction/CamoFormer-test'\
    --BR 'on'
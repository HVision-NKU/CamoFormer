pred_path=${1}
gt_path=${2}
name=${3}

python evaltools/eval.py   \
    --pred_root  $pred_path \
    --GT_root $gt_path \
    --model $name
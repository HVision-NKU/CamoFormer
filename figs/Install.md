```
conda create --name CamoFormer python=3.8.5
conda activate CamoFormer
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install opencv-python
conda install tensorboard
conda install tensorboardX
pip install timm
pip install matplotlib
pip install scipy
pip install einops

Please also install [apex](https://github.com/NVIDIA/apex).
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
or
Apex also support Python-only build (required with Pytorch 0.4):
pip install -v --no-cache-dir ./
```



# conda create -n mmcls python=3.8 -y; conda activate mmcls 이건 직접 해주세요..ㅠㅠ
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
git checkout jeom/KT
pip3 install -e .
pip install future tensorboard
pip install setuptools==59.5.0
pip install rich
mkdir data
# symbolic link for dataset : src -> dst
# ln -s ~/workspace/train data/food_dataset

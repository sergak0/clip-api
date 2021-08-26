# Clip api 
## Установка c GPU
### Install conda

https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

### Install CLIP
```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
### Install requirements.txt

`pip install -r requirements.txt`

### Run api

`nohup python3 main.py &`
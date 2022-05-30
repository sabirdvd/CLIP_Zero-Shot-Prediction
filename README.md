# CLIP Zero Shot Prediction 
 
Zero-Shot Prediction for 1000 imagenet classes 

For fast start [Colab](https://colab.research.google.com/drive/1bUOngE4T5GoyurxwDslprxfRRrigcX3J?usp=sharing) 

Download the mode from [openAI](https://github.com/openai/CLIP)

```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.1
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Base model ViT-B/32
Larage model: ViT-L/14

```
# flags 
--c imagenet classesss 
--s size of the model 
--i input image 
``` 

Large model # ViT-L/14 (Jan 2022 model) 
```
python CLIP_run.py --c imagenet_classes.txt    --s ViT-L/14 --i  /image/COCO_train2014_000000010881.jpg
```

## CLIP Zero Shot Prediction (Jap)
```
python 3.7
pip install git+https://github.com/rinnakk/japanese-clip.git
```
run 

```
python CLIP_run.py 
```


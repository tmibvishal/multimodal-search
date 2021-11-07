We are using the following image captioning
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning and we are also using a pretrained fastnet using
resnet backbone for object detection

Install all the required libraries

```
pip install -r requirements.txt
```

You need to download model checkpoint (pretrained) and the corresponding word_map here
https://drive.google.com/open?id=189VY65I_n4RTpQnmLGj7IzVnOF6dmePC
Download and paste 2 files in a directory `model_checkpoints`
Now run

```
python main.py'
```
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

Now, download flickr8k.zip dataset. Extract the files in data folder and overwrite the existing files
I have removed data/flickr8k/Images but kept other files. Make sure those Images are in that folder

Pre generate all text queries for all images. Makes evaluation easy
```
python gen_queries.py -i data/flickr8k/image_caption.txt -q data/queries_8k.pkl
```

Now run evaluation
image_caption_val.txt contains only 117 documents
```
python evaluation.py -i data/flickr8k/image_caption.txt -c data/collection_saved.pkl -q data/queries_8k.pkl -o output/output_bm25 -f bm25
```
Evaluation will be saved in temporary_directory/evaluation.txt

# Toxic-comments-detection

<p align="center">
  <img src="demo_image.png" width="600" title="Demo image">
</p>

This is a deployable machine learning model to rate toxic levels of comments from social networks. The data is taken from [Kaggle toxic comments classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Required packages
```bash
pip install -r requirements.txt
```

## Training
```bash
python src/train.py
```

The basic idea for this model:
* Clean text: lower all text, remove uncommon signs, expand abbreviations
* Tokenizing text data
* Create embedding vector using [Glove.6B](https://nlp.stanford.edu/projects/glove/)
* Train a deeplearning network with a bidirectional LSTM layer followed by two fully connected layers.

## Building application
```bash
python src/app_toxic_comments.py
```
The app is built using Flask api.

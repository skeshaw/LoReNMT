#!/bin/bash

# download Gujarati fastText embeddings
curl -o wiki.gu.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.gu.vec

# prepare German embeddings from pretrained de-en model
cd ../../fairseq
python extract_de_embed.py

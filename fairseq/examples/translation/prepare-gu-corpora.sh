#!/bin/bash

mkdir tmp

# download files
curl -o tmp/wikipedia.gu-en.tsv.gz http://data.statmt.org/wmt19/translation-task/wikipedia.gu-en.tsv.gz
curl -o tmp/wikititles.gu-en.tsv.gz http://data.statmt.org/wikititles/v1/wikititles-v1.gu-en.tsv.gz
curl -o tmp/govin.gu-en.tsv.gz  http://data.statmt.org/wmt19/translation-task/govin-clean.gu-en.tsv.gz

# unzip
cd tmp
gzip -d *tsv.gz
cd ..

# install langdetect
pip install langdetect

# clean corpora
python clean_gu_corpora.py
rm -rf tmp

# separate cleaned data by language
awk -F '|' '{print $1}' train.tags.gu-en.tmp > train.tags.gu-en.tmp.gu
awk -F '|' '{print $2}' train.tags.gu-en.tmp > train.tags.gu-en.tmp.en

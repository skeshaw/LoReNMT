#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt
BPE_TOKENS=20000

URLS=(
    "https://wit3.fbk.eu/download.php?release=2018-01&type=texts&slang=eu&tlang=en"
)
FILES=(
    "gu-en.gz"
)
CORPORA=(
    "gu-en"
)
RAW=(
    "train.tags.gu-en.tmp"
)

# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
OUTDIR=wmt19_gu_en/baseline

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=gu
tgt=en
lang=gu-en
prep=$OUTDIR
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig

for l in $src $tgt; do
    for f in "${RAW[@]}"; do
        awk '{gsub(/\"/,"")};1' $f.$l > train.tags.$lang.$l
    done
done

cd ..


echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in train.tags.$lang; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done


# echo "pre-processing test data..."
# for l in $src $tgt; do
#     if [ "$l" == "$src" ]; then
#         t="src"
#     else
#         t="ref"
#     fi
#     grep '<seg id' $orig/test/newstest2016-deen-$t.$l.sgm | \
#         sed -e 's/<seg id="[0-9]*">\s*//g' | \
#         sed -e 's/\s*<\/seg>\s*//g' | \
#         sed -e "s/\’/\'/g" | \
#     perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
#     echo ""
# done


echo "splitting train, valid and test..."
for l in $src $tgt; do
    awk '{if (NR%20 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/test.$l
    awk '{if (NR%10 == 0 && NR%20 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%10 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done



TRAIN=$tmp/train.eu-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done
for L in $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done



perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done

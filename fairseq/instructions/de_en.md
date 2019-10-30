### 1. Preprocess data
```
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref examples/translation/wmt18_en_de/train \
  --validpref examples/translation/wmt18_en_de/valid \
  --testpref examples/translation/wmt18_en_de/test \
  --destdir data-bin/wmt18_de_en
```

### 2. Train model
```
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/wmt18_de_en/ \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-07 --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --best-checkpoint-metric ppl --save-dir /home/ckpts
```

### 3. Generate predictions
- Generate outputs
```
fairseq-generate data-bin/wmt18_de_en \
    --path /home/ckpts/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    | tee /tmp/en_gen.out
```

- Write to source and reference files
```
grep ^H /tmp/en_gen.out | cut -f3- > en_gen.out.sys
grep ^T /tmp/en_gen.out | cut -f2- > en_gen.out.ref
```

- Get BLEU scores
```
fairseq-score --sys en_gen.out.sys --ref en_gen.out.ref --ignore-case
```

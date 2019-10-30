### 1. Preprocess data
- Differs only when using a target dictionary from a pretrained model for a different pair, say, de-en  
```
fairseq-preprocess --source-lang eu --target-lang en \
    --trainpref examples/translation/iwslt18_eu_en/transfer_fixed_target/train \
    --validpref examples/translation/iwslt18_eu_en/transfer_fixed_target/valid \
    --testpref examples/translation/iwslt18_eu_en/transfer_fixed_target/test \
    --tgtdict examples/translation/dict.en.txt \
    --destdir data-bin/iwslt18_eu_en/tranfer_fixed_target
```

### 2. Train model

#### a. Baseline
```
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt18_eu_en/baseline \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --max-epoch 50\
    --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt \
    --dropout 0.3 --weight-decay 1e-4 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --decoder-embed-path /home/wiki.en.vec \
    --save-dir /home/ckpts/baseline
```

#### b. Transfer
```
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt18_eu_en/transfer \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --max-epoch 36\
    --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt \
    --dropout 0.3 --weight-decay 1e-4 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --save-dir /home/ckpts/transfer --reset-optimizer
```
- Upgrade state_dict in encoder and decoder of `transformer.py`, since using different vocabs than the pretrained model. Embeddings either start from scratch or use pretrained vectors.
-- Refer to [this issue](https://github.com/pytorch/fairseq/issues/1196).


#### c. Transfer - fixed target
```
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt18_eu_en/tranfer_fixed_target/ \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --max-epoch 56\
    --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt \
    --dropout 0.3 --weight-decay 1e-4 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --encoder-embed-path /home/wiki.eu.vec \
    --save-dir /home/ckpts/transfer_fixed_target_with_embed/ --reset-optimizer
```
- Only upgrade state_dict in encoder of `transformer.py`, since target side is fixed

#### d. Transfer - fixed target with crosslingual mapping
- Same instruction as (c), but need to load mapping matrix in `upgrade_state_dict` in `transformer.py`

### 3. Generate predictions
- Same as in other cases


## Working with a different language pair  
Follow the same instructions as outlined here for any other language pair, like gu-en. The only changes that need to be made lie in `examples/translation` - once data is preprocessed, the rest of the process remains identical.

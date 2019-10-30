-- Ensure that `fairseq` and `faiss` are installed before running the following processes.  

### 1. Preprocess
- Go to the folder `data`.
- Run the bash script `prep_embed.sh`.

### 2. Train  
Run the following command to train, while in the home directory (`/MUSE`) -  
```
python unsupervised.py --src_lang gu --tgt_lang de --src_emb /data/wiki.gu.vec --tgt_emb data/nmt.de.vec \
  --exp_path data --exp_name gu_de_map --dico_max_rank 10000 \
  --dis_most_frequent 20000 --adversarial True --n_epochs 2 \
  --n_refinement 10
```

For detailed instructions regarding training and setting parameters, refer to the original `README.md`.

** The mapping will be saved under the name `best_mapping.pth` by default, within a folder inside `data/gu_de_map`.

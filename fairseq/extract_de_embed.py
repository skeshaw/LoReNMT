# after fairseq is installed in editable mode
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

# load embedding weights in a tensor
model_path = '../data/checkpoint_best.pt'
pretrained_model = load_checkpoint_to_cpu(model_path)
de_embed_tensor = pretrained_model['model']['encoder.embed_tokens.weight']

# save embeddings to a file
de_embed_path = '../MUSE/data/nmt.de.vec'

with open(de_embed_path, 'w') as fp:
    num_words, embed_dim = de_embed_tensor.shape
    # write size as the first line
    fp.write(' '.join(map(str, de_embed_tensor.shape)) + '\n')

    de_embed_list = de_embed_tensor.data.tolist()
    for i in range(num_words):
        embed = de_embed_list[i]
        word = 'random' + str(i)
        fp.write('{} {}\n'.format(word, ' '.join(map(str, embed))))

        if i%1000==0:
            print('Finished processing {} words'.format(i))

fp.close()

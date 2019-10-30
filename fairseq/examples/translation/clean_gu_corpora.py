import pandas as pd
from langdetect import detect
from glob import glob

def write_to_file(df):
    outfile = 'train.tags.gu-en.tmp'

    fp = open(outfile, 'a')

    for i, row in df.iterrows():
        fp.write('|'.join(row[:2]))
        fp.write('\n')
    fp.close()
    return

def filter_fn(row):
    try:
        return detect(row[0])
    except:
        return ''

for fname in glob('tmp/*.tsv'):
    text_df = pd.read_csv(fname, sep='\t', header=None, error_bad_lines=False).\
      dropna().drop_duplicates()
    text_df['lang_flag'] = text_df.apply(filter_fn, axis=1)
    filtered_df = text_df[text_df.lang_flag=='gu']
    write_to_file(filtered_df)

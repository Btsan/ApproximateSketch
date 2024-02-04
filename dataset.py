
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from bert import Tokenizer

def discretize(data):
    value_counts = data.value_counts(dropna=False)
    value_counts.sort_index(inplace=True, na_position='first')
    dvs = value_counts.index.values

    is_nan = pd.isnull(dvs)
    if is_nan.any():
        assert is_nan.sum() == 1, is_nan
        no_nan = dvs[~is_nan]
        cat = pd.Categorical(data, categories=no_nan)
        bin_ids = cat.codes + 1
        discrete_map = {float('nan'): 0}
        for i, val in enumerate(no_nan):
            discrete_map[val] = i + 1
    else:
        cat = pd.Categorical(data, categories=dvs)
        bin_ids = cat.codes
        discrete_map = {val: i for i, val in enumerate(dvs)}
    bin_ids = bin_ids.astype(np.int32, copy=False)
    return bin_ids, discrete_map, value_counts

def get_dataframe(filepath, names, columns):
    df = pd.read_csv(filepath,
                     names=names,
                     usecols=columns,
                     escapechar='\\',
                     encoding='utf-8',
                     on_bad_lines='skip',
                     low_memory=False)[columns]
    return df

class TableDataset(Dataset):
    def __init__(self, dataframe, intervals, bin_hashes, one_hashes, table_id=0, max_length=None, granularity={}):
        ''' dataset for a table
            - dataframe: a pandas dataframe
            - intervals: dict mapping each column to a list of interval sizes
            - table_id: int id of this dataframe
        '''
        super(TableDataset, self).__init__()
        
        self.table_id = table_id
        self.table = dataframe
        self.intervals = intervals
        self.bin_hash = bin_hashes
        self.one_hash = one_hashes

        # discretize tables and save the vocab mapping, counts
        self.vocab = {}
        self.counts = {}
        for col in self.table.columns:
            self.table[col], self.vocab[col], self.counts[col] = discretize(self.table[col])

        self.tokenizer = Tokenizer(self.table_id,
                                   self.table.columns,
                                   self.intervals,
                                   self.bin_hash,
                                   self.one_hash,
                                   max_length=max_length,
                                   granularity=granularity,)
    
        self.tokenizer.set_vocab(self.vocab, self.counts, dataframe.shape[0])
        
        # column masking probability
        self.column_mask = 0.15
        self.mask = 0.35

    def stats(self):
        for k,v in self.vocab.items():
            print(f'{k}: {len(v)}')

    def sketch(self, key, predicates={}):
        query = []
        for col, preds in predicates.items():
            if '=' in preds:
                assert preds['='] in self.vocab[col], f"equality predicate `{col}={preds['=']}` doesn't exist in vocab"
                val = self.vocab[col][preds['=']]
                p = f"`{col}` == {val}"
            else:
                assert '<' in preds or '>' in preds, f"missing valid predicate operator for {col}: {preds}"
                if '<' in preds:
                    val = self.tokenizer.leq(preds['<'], self.vocab[col])
                    p = f"`{col}` < {val}"
                else:
                    val = self.tokenizer.geq(preds['>'], self.vocab[col])
                    p = f"`{col}` > {val}"
            query.append(p)
        query = ' and '.join(query)
        res = self.table.query(query) if query else self.table
        counts = res.groupby([key]).size()

        counts = torch.tensor(np.array([counts.index.values, counts.values]), dtype=torch.int64).t()
        bins = torch.stack([b_hash(counts[:, 0]) for b_hash in self.bin_hash])
        ones = torch.stack([o_hash(counts[:, 0]) for o_hash in self.one_hash]).float()
        counts = counts.float()

        sketch_shape = (len(self.bin_hash), self.bin_hash[0].num_buckets)
        pos = torch.zeros(sketch_shape, dtype=torch.float32)
        neg = torch.zeros(sketch_shape, dtype=torch.float32)
        
        pos.scatter_(-1, bins, ones * (ones > 0) * counts[:, 1], reduce='add') # positive values
        neg.scatter_(-1, bins, ones * (ones < 0) * counts[:, 1], reduce='add') # negative values

        return pos, neg, res.shape[0]

    def __getitem__(self, index):
        record = self.table.iloc[index]
        return self.tokenizer(record, column_mask=self.column_mask, pos_mask=self.mask)

    def __len__(self):
        return len(self.table)
    
def get_table_dataset(filepath, names, columns, intervals, bin_hashes, one_hashes, table_id=0, max_length=None, granularity={}):
    df = pd.read_csv(filepath,
                     names=names,
                     usecols=columns,
                     escapechar='\\',
                     encoding='utf-8',
                     on_bad_lines='skip',
                     low_memory=False)[columns]
    dataset = TableDataset(df,
                           intervals,
                           bin_hashes,
                           one_hashes,
                           table_id=table_id,
                           max_length=max_length,
                           granularity=granularity)
    return dataset
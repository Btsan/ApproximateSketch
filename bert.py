from itertools import product, chain
from bisect import bisect_left, bisect_right

import torch
from torch.nn.functional import normalize
import numpy as np
from transformers import BertConfig, BertForMaskedLM

from xi import Xi, B_Xi

def get_bert(num_tables=1, num_hashes=5, num_bins=64, num_heads=12, num_layers=12, embed_size=768, feedforward_size=3072, max_length=512, pretrained=None, fp16=False):
    ''' returns a bert model with a suitable number of embeddings
    '''
    if pretrained:
        return BertForMaskedLM.from_pretrained(pretrained, torch_dtype=torch.float16 if fp16 else None,)
    num_embeddings = num_bins * 2 # removed: `num_tables`, `num_hashes`` due to type_vocab_size
    config = BertConfig(
        vocab_size=num_embeddings + 1, # add 1 to reserve 0th embedding for MASK token
        hidden_size=embed_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=feedforward_size,
        max_position_embeddings=max_length,
        type_vocab_size=num_tables * num_hashes,
        torch_dtype='float16' if fp16 else None,
    )
    bert = BertForMaskedLM(config)
    return bert
    
class Tokenizer:
    def __init__(self, table_id, columns, interval_sizes, bin_hashes, one_hashes, max_length=512, granularity={}):
        ''' Tokenizer factorizes and hashes a specific table's records for input to bert
            - table_id: integer id of table which this tokenizer is associated with
                - used to 
            - columns: list of column names in the table e.g., a dataframe
                - factorized records will use this column order
            - interval_sizes: dict mapping column names to lists containing interval sizes
                - if column name isn't specified, uses default interval size 1
            - bin_hashes: list of calleable hash functions that map to bin ids
            - one_hashes: list of calleable hash functons that map to +-1
            - max_length: int. pads tokenized inputs up to this size
        '''
        assert len(bin_hashes) == len(one_hashes)
        
        self.table_id = table_id
        self.columns = tuple(columns)
        self.bin_hash = tuple(bin_hashes)
        self.one_hash = tuple(one_hashes)
        self.max_length = max_length
        self.granularity = granularity

        # save intervals sizes
        # if column doesn't have specified intervals, default to interval size 1
        self.interval_sizes = {col: (tuple(interval_sizes[col]) if col in interval_sizes else (1,)) for col in columns}

        # constant size of tokenized records (not including padding)
        self.size = sum(map(len, self.interval_sizes.values())) * len(self.bin_hash)

        # constant vectors
        self.token_type_ids = torch.zeros(self.max_length, dtype=torch.int64) + self.table_id
        self.attention_mask = torch.zeros(self.max_length, dtype=torch.int64)
        self.attention_mask[:self.size] = 1

        # special token embedding constant indices
        self.MASK = 0

        # table specific data
        # must be initialized by calling self.set_vocab(dict, dict, int)
        self.vocab = None
        self.counts = None
        self.num_tuples = 1

        # names of columns and their derived intervals (before hashing)
        self.names = []
        for col, sizes in zip(columns, self.interval_sizes):
            self.names += [f'{col}//{s}' for s in sizes]

    def set_vocab(self, vocab, counts, num_tuples=1):
        ''' must be called before sketching selection
            - vocab: dict of dicts mapping values to discrete ids per column {val any: id int}
            - counts: dict of dataframes indicating count of values per column e.g., column.value_counts()
            - num_tuples: integer number of tuples in the table associated with the given counts
        '''
        self.vocab = vocab
        self.counts = counts
        self.num_tuples = num_tuples

    def decode(self, logits, key, weights=None):
        ''' returns sketch from the given logits and mask
            - logits: tensor shaped (seq_len, vocab) of raw logits
            - record: list of hashable values, table record
            - mask: list containing positions of masked tokens
            - table: str name of table that has the record
            - key: int denoting position of the key column to sketch
        '''
        sketches = []
        key_pos = 0
        for col in self.columns:
            if col == key:
                break
            key_pos += len(self.interval_sizes[col])
        seq_size = sum(map(len, self.interval_sizes.values()))

        rows = []
        for i in range(len(self.bin_hash)):
            sequence_start = seq_size * i
            rows.append(logits[:, sequence_start + key_pos, 1:])
            # print(f'row {i} at {sequence_start}+{key_pos}')
        rows = torch.stack(rows, dim=1)
        # rows = rows.softmax(dim=2)
        rows = normalize(rows - rows.min(dim=2).values.unsqueeze(2), p=1, dim=2)
        if weights != None:
            assert len(weights) == logits.shape[0], f'number of weights ({len(weights)}) must equal logits batch size ({logits.shape}) in dim 0'
            weights = torch.tensor(weights)
            if weights.dim() == 1:
                weights = weights.unsqueeze(1).unsqueeze(2)
            rows *= weights 
        sketches = rows.sum(dim=0)

        return sketches, rows

    def sketch(self, predicates, key):
        pass

    def decompose_range(self, a, b, sizes, col, sampling_limit=-1, granularity=0, limit=-1):
        ''' decomposes a range [a, b] into all of its intervals and subintervals that intersects it
            - a: left bound of range (inclusive)
            - b: right bound of range (also inclusive)
            - sizes: list of interval sizes
            returns list of tuples of intervals

            preds =     10 >= a >= 1, 5 >= b >= 1
            sizes =     a:[8, 1] b:[16, 2]
            intervals = a:[0, 1] b:[ 0, 0]
                              2         1
                              3         1
                              4         2
                              5         2
                              6
                              7
                          [1, 8]
                              9
                              10
            sizes =     a:[1, 8] b:[2, 16]
            intervals = a:[1, 0] b:[0, 0]
                          [2, 0]   [1, 0]
                          [3, 0]   [1, 0]
                          [4, 0]   [2, 0]
                          [5, 0]   [2, 0]
                          [6, 0]
                          [7, 0]
                          [8, 1]
                          [9, 1]
                          [10, 1]
            the process is the same regardless of the order of interval sizes 
        '''

        # automatic granularity
        granularity = max(sizes)
        for s in sorted(sizes):
            if (b - a + 1) // s <= sampling_limit:
                granularity = s
                break 

        ascending = sorted([s for s in sizes if s >= granularity])
        if len(ascending) == 0:
            intervals = [(self.MASK,) * len(sizes)]
            weights = [(self.num_tuples,) * len(sizes)]
        smallest = min(ascending)
        intervals = []
        weights = []
        
        # sampling weight, take only the n-th most probable intervals
        p = []

        # print(f'{col}: for i in range({a}, {b}+1, {smallest}):')
        for i in range(a, b+1, smallest):
            intervals.append(tuple([i // s if s >= granularity else self.MASK for s in sizes]))
            # count = self.counts[col].iloc[i] # previous incorrect weight
            count = self.counts[col].iloc[(i // smallest) * smallest : (i // smallest) * (smallest + 1)].sum()
            weights.append(tuple([count] * len(sizes)))
            p.append(count)

        # sort intervals and weights in descending order of p
        sorted_intervals = []
        sorted_weights = []
        for _, interval, weight in sorted(zip(p, intervals, weights)):
            sorted_intervals.append(interval)
            sorted_weights.append(weight)

        # take only the n-th most probable. n=sampling_limit
        return sorted_intervals[:sampling_limit], sorted_weights[:sampling_limit]

    def decompose_range_v2(self, a, b, sizes, col, sampling_limit=-1, granularity=0, limit=-1):
        ascending = sorted(sizes)
        intervals = []
        weights = []
        p = []

        i = a
        while i < b:
            # print(f'Decomposing range [{i}, {b}]')
            excess = size = max(sizes)
            for s in ascending:
                left_excess = i - (i // s) * s
                right_excess = (i // s + 1) * s - b
                total_excess = left_excess + (right_excess if right_excess > 0 else 0)
                if total_excess <= excess:
                    excess = total_excess
                    size = s
                # print(f'\t testing interval size {s} [{(i // s) * s}, {(i // s + 1) * s}]: excess {total_excess}')
            left = (i // size) * size
            left = left if left >= a else a
            right = (i // size + 1) * size
            right = right if right <= b else b
            count = self.counts[col].iloc[left:right].sum()
            if count == 0:
                i = (i // size + 1) * size
                continue
            intervals.append(tuple([i // s if s >= size else self.MASK for s in sizes]))
            weights.append(tuple([count] * len(sizes)))
            p.append(count)
            # print(f'Selected interval size {size} which contains {self.counts[col].iloc[(i // size) * size:(i // size + 1) * size].sum()}/{count} records')
            i = (i // size + 1) * size
        
        if len(intervals) == 0:
            # print('Returning 0 intervals')
            intervals = [(self.MASK,) * len(sizes)]
            weights = [(self.num_tuples,) * len(sizes)]
            return intervals, weights

        sorted_intervals = []
        sorted_weights = []
        for _, interval, weight in sorted(zip(p, intervals, weights)):
            sorted_intervals.append(interval)
            sorted_weights.append(weight)
            
        # print(f'Returning top {len(sorted_intervals[:sampling_limit])} intervals which contains {sum(p[:sampling_limit])}/{self.counts[col].iloc[a:b].sum()} records')
        return sorted_intervals[:sampling_limit], sorted_weights[:sampling_limit]
    
    def decompose_range_v3(self, a, b, sizes, col, sampling_limit=-1, granularity=0, limit=-1):
        descending = sorted(sizes, reverse=True)
        intervals = []
        weights = []
        p = []

        i = a
        while i < b:
            # print(f'Decomposing range [{i}, {b}]')
            noise = self.num_tuples
            size = max(sizes)
            count = 0
            for s in descending:
                total_rows = self.counts[col].iloc[(i // s) * s:(i // s + 1) * s].sum()
                left = (i // s) * s
                left = left if left >= a else a
                right = (i // s + 1) * s
                right = right if right <= b else b
                relevant_rows = self.counts[col].iloc[left:right].sum()
                irrelevant_rows = total_rows - relevant_rows
                if irrelevant_rows < noise:
                    noise = irrelevant_rows
                    size = s
                    count = relevant_rows
                # print(f'\t testing interval size {s} [{(i // s) * s}, {(i // s + 1) * s}]: noise {irrelevant_rows}')
            if count == 0:
                i = (i // size + 1) * size
                continue
            intervals.append(tuple([i // s if s >= size else self.MASK for s in sizes]))
            weights.append(tuple([count] * len(sizes)))
            p.append(count)
            # print(f'Selected interval size {size} which contains {noise + count}/{count} relevant records')
            i = (i // size + 1) * size
        
        if len(intervals) == 0:
            # print('Returning 0 intervals')
            intervals = [(self.MASK,) * len(sizes)]
            weights = [(self.num_tuples,) * len(sizes)]
            return intervals, weights
        
        sorted_intervals = []
        sorted_weights = []
        for _, interval, weight in sorted(zip(p, intervals, weights)):
            sorted_intervals.append(interval)
            sorted_weights.append(weight)
            
        # print(f'Returning top {len(sorted_intervals[:sampling_limit])} intervals')
        return sorted_intervals[:sampling_limit], sorted_weights[:sampling_limit]
    
    def decompose_range_v4(self, a, b, sizes, col, sampling_limit=-1, granularity=0, limit=-1):
        descending = sorted(sizes, reverse=True)
        intervals = []
        weights = []
        p = []

        # first cover [a, b] with largest interval size
        i = a
        while i < b:
            s = max(sizes)
            total_rows = self.counts[col].iloc[(i // s) * s:(i // s + 1) * s].sum()
            left = (i // s) * s
            left = left if left > a else a
            right = (i // s + 1) * s
            right = right if right < b else b
            intersection = self.counts[col].iloc[left:right].sum()
            difference = total_rows - intersection
            if intersection > 0:
                intervals.append({'left': (i // s) * s, 'right': (i // s + 1) * s, 'intersection': intersection, 'difference': difference, 'i': i, 'size': s})
            i = (i // s + 1) * s
            print(f'cover [{(i // s) * s}, {(i // s + 1) * s}] which has a difference = {difference}')

        # then iteratively refine edges
        for s in descending[1:]:
            for candidate in [x for x in range(len(intervals)) if x == 0 or x == (len(intervals)-1)]:
                if intervals[candidate]['difference'] == 0:
                    continue
                replacements = []
                i = max(intervals[candidate]['left'], a)
                while i < min(intervals[candidate]['right'], b):
                    total_rows = self.counts[col].iloc[(i // s) * s:(i // s + 1) * s].sum()
                    left = (i // s) * s
                    left = left if left > a else a
                    right = (i // s + 1) * s
                    right = right if right < b else b
                    intersection = self.counts[col].iloc[left:right].sum()
                    difference = total_rows - intersection
                    if intersection > 0 and difference < intervals[candidate]['difference']:
                        replacements.append({'left': (i // s) * s, 'right': (i // s + 1) * s, 'intersection': intersection, 'difference': difference, 'i': i, 'size': s})
                    i = (i // s + 1) * s
                if replacements:
                    print(f'replace [{intervals[candidate]["left"]}, {intervals[candidate]["right"]}] difference = {intervals[candidate]["difference"]}')
                    for x in replacements:
                        print(f'\t\\[{x["left"]}, {x["right"]}] difference = {x["difference"]}')
                    if candidate == 0:
                        intervals = replacements + intervals[1:]
                    else:
                        intervals = intervals[:-1] + replacements
                        
        # convert intervals into expected format
        interval_ids = []
        for interval in intervals:
            interval_ids.append(tuple([interval['i'] // s if s >= interval['size'] else self.MASK for s in sizes]))
            weights.append(tuple([interval['intersection']] * len(sizes)))
            p.append(interval['intersection'])

        # sort and return most relevant intervals
        sorted_intervals = []
        sorted_weights = []
        for _, interval, weight in sorted(zip(p, interval_ids, weights)):
            sorted_intervals.append(interval)
            sorted_weights.append(weight)
        print(f'returning {len(sorted_intervals)}/{len(interval_ids)} intervals')
        return sorted_intervals[:sampling_limit], sorted_weights[:sampling_limit]
    
    def leq(self, val, vocab):
        '''
            a b c d e f g
            0 1 2 3 4 5 6

            x <= e
            4 
        '''
        keys = list(vocab.keys())
        pos = bisect_right(keys, val)
        return vocab[keys[pos]]
    
    def geq(self, val, vocab):
        '''
            a b c d e f g
            0 1 2 3 4 5 6

            x >= b
            1
        '''
        keys = list(vocab.keys())
        pos = bisect_left(keys, val)
        return vocab[keys[pos]]

    def sketch_upper_bound(self, predicates, test_mask=False, decompose_limit=-1):
        ''' Prepare model input to estimate the distribution of the sketch
            Also a list of weights to determine the mass given to each bucket
        '''
        assert self.vocab, f'self.vocab is {self.vocab}. Must be set prior to sketching e.g., self.set_vocab(dict)'
        
        intervals = []
        mask = []
        weights = []
        weights2 = []

        # first factorize the predicate into subselections
        pos = 0
        for col in self.columns:
            sizes = self.interval_sizes[col]
            vocab = self.vocab[col]
            if col in predicates:
                preds = predicates[col]
                if '=' in preds:
                    assert preds['='] in vocab, f"equality predicate `{col}={preds['=']}` doesn't exist in vocab"
                    val = vocab[preds['=']]
                    intervals += [[(val // s,)] for s in sizes]

                    count = self.counts[col].iloc[val]
                    weights += [[(count,)]] * len(sizes)
                    weights2.append(count)
                else:
                    assert '<' in preds or '>' in preds, f'No acceptable operators (`=`, `<`, `>`) in predicate for {col}. {preds.keys()}'
                    low = self.geq(preds['>'], vocab) if '>' in preds else min(vocab.values())
                    high = self.leq(preds['<'], vocab) if '<' in preds else max(vocab.values())
                    # intervals.append(self.decompose_range(low, high, sizes))
                    count = self.counts[col].iloc[low:high].sum()
                    subintervals, counts = self.decompose_range_v4(low,
                                                                high,
                                                                sizes,
                                                                col,
                                                                sampling_limit=decompose_limit,
                                                                granularity=self.granularity[col] if col in self.granularity else 0)
                    intervals.append(subintervals)
                    weights.append(counts)
                    weights2.append(count)
            else:
                mask += [pos + i for i in range(len(sizes))]
                intervals += [[(self.MASK,)]] * len(sizes)
                weights += [[(self.num_tuples,)]] * len(sizes)
                weights2.append(self.num_tuples)
            pos += len(sizes)

        batch_intervals = [tuple(chain(*prod)) for prod in product(*intervals)]
        batch_intervals = torch.tensor(batch_intervals, dtype=torch.int64)
        assert batch_intervals.shape[-1] == self.size / len(self.bin_hash), batch_intervals.shape
        # print('batch_intervals', batch_intervals, batch_intervals.shape)
        # return batch_intervals

        batch_weights = [tuple(chain(*prod)) for prod in product(*weights)]
        batch_weights = torch.tensor(batch_weights, dtype=torch.float32) #[b, seqsize]
        # print('batch_weights', batch_weights, batch_weights.shape)

        hashes = []
        token_types = []
        for i, (b_hash, o_hash) in enumerate(zip(self.bin_hash, self.one_hash)):
            bins = b_hash(batch_intervals) + 1 # 0th embedding is mask
            ones = o_hash(batch_intervals)
            bins[ones < 0] += b_hash.num_buckets
            bins[:, mask] = self.MASK
            hashes.append(bins)
            
            # token type for this table and hash pair
            token_types += [self.table_id * len(self.bin_hash) + i] * batch_intervals.shape[-1]
        input_ids = torch.cat(hashes, axis=-1)
        # return input_ids, batch_intervals, mask, batch_weights
        # print('input_ids', input_ids, input_ids.shape)
        input_ids, inverse = input_ids.unique(dim=0, return_inverse=True)
        # print('inverse', inverse, inverse.shape)
        input_weights = batch_weights.min(dim=1)[0].gather(0, inverse)
        # print('input_weights', input_weights)
        input_weights = normalize(input_weights, p=1, dim=0) * min(weights2)
        token_type_ids = torch.tensor([token_types] * input_ids.shape[0], dtype=torch.int64) 
        if test_mask:
            attention_mask = torch.ones((1, batch_intervals.shape[-1]), dtype=torch.float32)
            attention_mask[:, mask] = 0
            attention_mask = attention_mask.repeat(input_ids.shape[0], len(self.bin_hash))
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        
        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask}, input_weights

    def __call__(self, record, column_mask=0.0, pos_mask=0.0):
        ''' creates inputs to train bert to predict masked tokens
            returns 3 vectors: input_ids, token_type_ids, and attention_mask
            - record: dict or pandas Series of the table record e.g., a dataframe's row
            - column_mask: float probability of masking an entiretitle column
        '''
        input_ids = torch.zeros(self.max_length, dtype=torch.int64)
        labels = torch.zeros(self.max_length, dtype=torch.int64)
        token_type_ids = torch.zeros(self.max_length, dtype=torch.int64)
        
        intervals = []
        mask = []
        for col in self.columns:
            if np.random.rand() < column_mask:
                # whole column mask
                mask += [len(intervals) + i for i in range(len(self.interval_sizes[col]))]
            intervals += [(record[col] // size) for size in self.interval_sizes[col]]
        intervals = torch.tensor(intervals, dtype=torch.int64)

        hashes = []
        no_mask = []
        token_types = []
        for i, (b_hash, o_hash) in enumerate(zip(self.bin_hash, self.one_hash)):
            bins = b_hash(intervals) + 1 # 0th reserved by mask
            ones = o_hash(intervals)
            bins[ones < 0] += b_hash.num_buckets
            no_mask.append(bins.detach().clone())
            no_mask[-1][[pos for pos in range(len(bins)) if pos not in mask]] = -100 # loss not computed for label = -100
            bins[mask] = self.MASK
            hashes.append(bins)
            token_types += [self.table_id * len(self.bin_hash) + i] * len(intervals)
        input_ids[:self.size] = torch.cat(hashes, axis=-1)
        labels[:self.size] = torch.cat(no_mask, axis=-1)
        token_type_ids[:self.size] = torch.tensor(token_types, dtype=torch.int64)
        
        # mask random positions
        if pos_mask:
            rand = np.random.rand(self.max_length)
            input_ids[rand < pos_mask] = self.MASK

        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': self.attention_mask,
                'labels': labels}
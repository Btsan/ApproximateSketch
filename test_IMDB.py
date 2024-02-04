import argparse
from datetime import datetime
from time import perf_counter_ns
from itertools import permutations

import torch
import pandas as pd
from numpy.random import RandomState

import xi
import bert
import dataset

def qerror(pred, targ):
    pred = max(pred, 1)
    targ = max(targ, 1)
    err = max(pred/targ, targ/pred)
    return err

def merge_sketches(sketches):
    """
    sketches - tensor of row sketches to be merged
    """
    indices = torch.argmin(sketches.abs(), dim=0)
    merge_sketch = torch.tensor([sketches[row, i] for i, row in enumerate(indices)])
    return merge_sketch

def estimate_join(models, predicates, keys, datasets, attention_mask=False, middle=None, decompose_limit=-1, device_batch_size=256):

    actual_sketch = {}
    joint_size = {}
    approx_sketch = {}
    upper_sketch = {}

    #count-min
    actual_count = {}
    approx_count = {}
    upper_count = {}

    # count-mean-min
    actual_cmm = {}
    approx_cmm = {}
    upper_cmm = {}

    sketch_time = 0
    model_time = 0
    for table, preds in predicates.items():
        key = keys[table]

        sketch_start = perf_counter_ns()
        pos, neg, card = datasets[table].sketch(key, preds)
        sketch_time += perf_counter_ns() - sketch_start

        # Fast-AGMS
        actual_sketch[table] = pos + neg
        # print(f'actual sketch {actual_sketch[table].shape} {table}: {preds}\n{actual_sketch[table]}')

        # Count-Min
        actual_count[table] = torch.cat((pos, neg), dim=1).abs()
        # print(f'actual count {actual_count[table].shape} {table}: {preds}\n{actual_count[table]}')

        num_bins = actual_count[table].shape[1]
        # Count-Mean-Median
        actual_cmm[table] = actual_count[table] - ((actual_count[table].mean(dim=1, keepdim=True) * num_bins - actual_count[table]) / (num_bins - 1))
        # print(f'actual cmm {actual_cmm[table].shape} {table}: {preds}\n{actual_cmm[table]}')

        if preds:
            inputs, weights = datasets[table].tokenizer.sketch_upper_bound(preds, test_mask=attention_mask, decompose_limit=decompose_limit)
            # print(inputs)
            model_start = perf_counter_ns() 
            outputs = models[table](**inputs)
            model_time += perf_counter_ns() - model_start
            approx, _ = datasets[table].tokenizer.decode(outputs.logits, key)
            sketch_size = approx.shape[1] // 2

            # Fast-AGMS
            approx = approx * card
            approx_sketch[table] = approx[:, :sketch_size] - approx[:, sketch_size:]
            # print(f'approx sketch {approx_sketch[table].shape} {table}: {preds}\n{approx_sketch[table].to(int)}')

            upper_approx, _ = datasets[table].tokenizer.decode(outputs.logits, key, weights=weights)
            upper_sketch[table] = upper_approx[:, :sketch_size] - upper_approx[:, sketch_size:]
            # print(f'upper sketch {upper_sketch[table].shape} {table}: {preds}\n{upper_sketch[table].to(int)}')

            # Count-Min
            approx_count[table] = approx
            # print(f'approx count {approx_count[table].shape} {table}: {preds}\n{approx_count[table].to(int)}')

            upper_count[table] = upper_approx
            # print(f'upper count {upper_count[table].shape} {table}: {preds}\n{upper_count[table].to(int)}')

            # Count-Mean-Median
            approx_cmm[table] = approx_count[table] - ((approx_count[table].mean(dim=1, keepdim=True) * num_bins - approx_count[table]) / (num_bins - 1))
            # print(f'approx count {approx_cmm[table].shape} {table}: {preds}\n{approx_cmm[table].to(int)}')

            upper_cmm[table] = upper_count[table] - ((upper_count[table].mean(dim=1, keepdim=True) * num_bins - upper_count[table]) / (num_bins - 1))
            # print(f'upper count {upper_cmm[table].shape} {table}: {preds}\n{upper_cmm[table].to(int)}')

        else:
            approx_sketch[table] = upper_sketch[table] = actual_sketch[table]
            approx_count[table] = upper_count[table] = actual_count[table]
            approx_cmm[table] = upper_cmm[table] = actual_cmm[table]

        # exact joint
        joint_size[table] = card

    # Fast-AGMS
    actual_med = 1
    approx_med = 1
    upper_med = 1

    # Fast-AGMS Same-Sign
    actual_sign = 0
    approx_sign = 0
    upper_sign = 0

    # Count-Min
    actual_min = 1
    approx_min = 1
    upper_min = 1

    # Count-Mean-Median
    actual_mm = 1
    approx_mm = 1
    upper_mm = 1

    # Count-Mean-Median Same-Sign
    actual_mm_sign = 0
    approx_mm_sign = 0
    upper_mm_sign = 0
    
    # joint
    joint_est = 1

    # join estimations
    for i, table in enumerate(predicates):
        # Fast-AGMS products
        actual_med *= actual_sketch[table]
        approx_med *= approx_sketch[table]
        upper_med *= upper_sketch[table]

        # Fast-AGMS Same-Sign products
        # track sign of locations and only keep products between same-sign terms
        actual_sign += actual_sketch[table].sign()
        approx_sign += approx_sketch[table].sign()
        upper_sign += upper_sketch[table].sign()

        # Count-Min products
        actual_min *= actual_count[table]
        approx_min *= approx_count[table]
        upper_min *= upper_count[table]

        # Count-Mean-Median
        actual_mm *= actual_cmm[table]
        approx_mm *= approx_cmm[table]
        upper_mm *= upper_cmm[table]

        # Count-Mean-Median Same-Sign products
        actual_mm_sign += actual_cmm[table].sign()
        approx_mm_sign += approx_cmm[table].sign()
        upper_mm_sign += upper_cmm[table].sign()

        # joint product
        joint_est *= joint_size[table]

    # The following is equivalent to default Fast-AGMS for 2 sketches
    actual_unbiased = (actual_med * (actual_sign.abs() == len(predicates))).abs() - \
        (actual_med * (actual_sign.abs() < len(predicates))).abs()
    approx_unbiased = (approx_med * (approx_sign.abs() == len(predicates))).abs() - \
        (approx_med * (approx_sign.abs() < len(predicates))).abs()
    upper_unbiased = (upper_med * (upper_sign.abs() == len(predicates))).abs() - \
        (upper_med * (upper_sign.abs() < len(predicates))).abs()

    # Same-Sign Only estimates
    actual_sign = (actual_med * (actual_sign.abs() == len(predicates))).abs().sum(dim=1).quantile(0.5)
    approx_sign = (approx_med * (approx_sign.abs() == len(predicates))).abs().sum(dim=1).quantile(0.5)
    upper_sign = (upper_med * (upper_sign.abs() == len(predicates))).abs().sum(dim=1).quantile(0.5)

    # Fast-AGMS estimates
    actual_med = actual_med.sum(dim=1).quantile(0.5)
    approx_med = approx_med.sum(dim=1).quantile(0.5)
    upper_med = upper_med.sum(dim=1).quantile(0.5)

    # Count-Min estimates
    actual_min = actual_min.sum(dim=1).min()
    approx_min = approx_min.sum(dim=1).min()
    upper_min = upper_min.sum(dim=1).min()

    
    if middle:
        # COMPASS Fast-AGMS
        actual_compass = []
        approx_compass = []
        upper_compass = []

        # COMPASS Count-Mean-Median
        actual_ccmm = []
        approx_ccmm = []
        upper_ccmm = []

        # COMPASS Merge permutations
        other_actual_sketch = [actual_sketch[t] for t in predicates if t != middle]
        other_approx_sketch = [approx_sketch[t] for t in predicates if t != middle]
        other_upper_sketch = [upper_sketch[t] for t in predicates if t != middle]

        other_actual_cmm = [actual_cmm[t] for t in predicates if t != middle]
        other_approx_cmm = [approx_cmm[t] for t in predicates if t != middle]
        other_upper_cmm = [upper_cmm[t] for t in predicates if t != middle]

        for indices in permutations([i for i in range(len(keys)-1)], len(keys)-1):
            merged_actual_sketch = merge_sketches(actual_sketch[middle][[indices]])
            merged_approx_sketch = merge_sketches(approx_sketch[middle][[indices]])
            merged_upper_sketch = merge_sketches(upper_sketch[middle][[indices]])

            merged_actual_cmm = merge_sketches(actual_cmm[middle][[indices]])
            merged_approx_cmm = merge_sketches(approx_cmm[middle][[indices]])
            merged_upper_cmm = merge_sketches(upper_cmm[middle][[indices]])

            assert len(indices) == len(other_actual_sketch), f'{indices} not equal to {len(other_actual_sketch)}'
            for row, idx in enumerate(indices):
                merged_actual_sketch *= other_actual_sketch[row][idx]
                merged_approx_sketch *= other_approx_sketch[row][idx]
                merged_upper_sketch *= other_upper_sketch[row][idx]

                merged_actual_cmm *= other_actual_cmm[row][idx]
                merged_approx_cmm *= other_approx_cmm[row][idx]
                merged_upper_cmm *= other_upper_cmm[row][idx]
            
            actual_compass.append(merged_actual_sketch.sum().abs())
            approx_compass.append(merged_approx_sketch.sum().abs())
            upper_compass.append(merged_upper_sketch.sum().abs())

            actual_ccmm.append(merged_actual_cmm.sum().abs())
            approx_ccmm.append(merged_approx_cmm.sum().abs())
            upper_ccmm.append(merged_upper_cmm.sum().abs())
        
        actual_compass = torch.tensor(actual_compass).quantile(0.5)
        approx_compass = torch.tensor(approx_compass).quantile(0.5)
        upper_compass = torch.tensor(upper_compass).quantile(0.5)

        actual_ccmm = torch.tensor(actual_ccmm).quantile(0.5)
        approx_ccmm = torch.tensor(approx_ccmm).quantile(0.5)
        upper_ccmm = torch.tensor(upper_ccmm).quantile(0.5)
    else:
        actual_compass = actual_unbiased.sum(dim=1).abs().quantile(0.5)
        approx_compass = approx_unbiased.sum(dim=1).abs().quantile(0.5)
        upper_compass = upper_unbiased.sum(dim=1).abs().quantile(0.5)

        actual_ccmm = actual_mm.sum(dim=1).abs().quantile(0.5)
        approx_ccmm = approx_mm.sum(dim=1).abs().quantile(0.5)
        upper_ccmm = upper_mm.sum(dim=1).abs().quantile(0.5)
    
    #   unbiased estimate
    #   tends towards being hugely negative as number of tables increases
    actual_unbiased = actual_unbiased.sum(dim=1).quantile(0.5)
    approx_unbiased = approx_unbiased.sum(dim=1).quantile(0.5)
    upper_unbiased = upper_unbiased.sum(dim=1).quantile(0.5)

    # Count-Mean-Median estimates w/ Same-Sign check
    actual_mm = (actual_mm * (actual_mm_sign.abs() == len(predicates))).abs().sum(dim=1).quantile(0.5)
    approx_mm = (approx_mm * (approx_mm_sign.abs() == len(predicates))).abs().sum(dim=1).quantile(0.5)
    upper_mm = (upper_mm * (upper_mm_sign.abs() == len(predicates))).abs().sum(dim=1).quantile(0.5)

    return {'Fast-AGMS': actual_med,
            'Approx. Fast-AGMS': approx_med,
            'Upper Fast-AGMS': upper_med,

            'Same-Sign': actual_sign,
            'Approx. Same-Sign': approx_sign,
            'Upper Same-Sign': upper_sign,

            'Same-Unbiased': actual_unbiased,
            'Approx. Same-Unbiased': approx_unbiased,
            'Upper Same-Unbiased': upper_unbiased,

            'Count-Min': actual_min,
            'Approx. Count-Min': approx_min,
            'Upper Count-Min': upper_min,

            'Count-Mean-Median': actual_mm,
            'Approx. Count-Mean-Median': approx_mm,
            'Upper Count-Mean-Median': upper_mm,
            
            'Joint Product': joint_est,
            
            'COMPASS': actual_compass,
            'Approx. COMPASS': approx_compass,
            'Upper COMPASS': upper_compass,

            'COMPASS CMM': actual_ccmm,
            'Approx. COMPASS CMM': approx_ccmm,
            'Upper COMPASS CMM': upper_ccmm,

            'Sketch Time': sketch_time,
            'Model Time': model_time,
            }

if __name__ == '__main__':
    # DO NOT CHANGE THIS SEED
    rs = RandomState(2**31-1)
    # The hash functions used to hash the training data depends on this seed
    
    # for debugging long sketch vectors
    # torch.set_printoptions(precision=1, threshold=1000000, linewidth=200000)
    torch.set_printoptions(profile='full')
    # torch.set_printoptions(profile='short')

    parser = argparse.ArgumentParser(description='evaluates approximate sketch models on IMDB tables')
    parser.add_argument('--num_hash', default=5, type=int, help='number of independent hash functions')
    parser.add_argument('--num_bins', default=4096, type=int, help='number of buckets per hash function')
    parser.add_argument('--csv', type=str, default='job_light_sub_query_with_star_join.sql.txt', help='CSV containing (queries, JOB-light ID, cardinality)')
    parser.add_argument('--writefile', default='out.csv', help='name of output csv file')
    parser.add_argument('--mask', action='store_true', help='mask columns without predicates during model inference')
    parser.add_argument('--fp16', action='store_true', help='use float16 if set')
    parser.add_argument('--decompose_limit', type=int, default=-1, help='number of intervals for decomposing predicate ranges (-1 for all available intervals)')
    parser.add_argument('--path', default='./', type=str, help='path to IMDB data directory containing all table CSV files')
    parser.add_argument('--title', default='BERT_5x4096/title__10c/', help='directory of saved model for table title')
    parser.add_argument('--movie_companies', default='BERT_5x4096/movie_companies__10c/', help='directory of saved model for table movie_companies')
    parser.add_argument('--movie_info_idx', default='BERT_5x4096/movie_info_idx__10c/', help='directory of saved model for table movie_info_idx')
    parser.add_argument('--movie_keyword', default='BERT_5x4096/movie_keyword__10c/', help='directory of saved model for table movie_keyword')
    parser.add_argument('--movie_info', default='BERT_5x4096/movie_info__10c/', help='directory of saved model for table movie_info')
    parser.add_argument('--cast_info', default='BERT_5x4096/cast_info__10c/', help='directory of saved model for table cast_info')
    parser.add_argument('--device', default='cpu', help='cpu or gpu for model inference device')
    args = parser.parse_args()
    print(args)
    
    # hash functions
    bin_hashes = [xi.B_Xi(args.num_bins, rs.randint(2**32), rs.randint(2**32)) for _ in range(args.num_hash)]
    one_hashes = [xi.Xi(rs.randint(2**32), rs.randint(2**32)) for _ in range(args.num_hash)]

    datasets = {}
    models = {}

    ts = datetime.now()
    title = {'filepath': f'{args.path}/IMDB/data/title.csv',
             'names': ['id', 'title', 'imdb_index', 'kind_id', 'production_year',
                       'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr',
                       'episode_nr', 'series_years', 'md5sum'],
             'columns': ['id', 'kind_id', 'series_years', 'production_year',
                         'phonetic_code', 'season_nr', 'episode_nr', 'imdb_index'],
             'intervals': {'series_years': [8, 64, 256],
                           'production_year': [2, 16],
                           'phonetic_code': [32, 256, 2048],
                           'season_nr': [2, 16],
                           'episode_nr': [16, 128, 1024],
                           'imdb_index': [2]},
             'granularity': {'series_years': 256,
                           'production_year': 16,
                           'phonetic_code': 256,
                           'season_nr': 16,
                           'episode_nr': 128,},
             'table_id': 0}
    title_dataset = dataset.get_table_dataset(**title, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=128)
    delta = datetime.now() - ts
    print(f"Loaded {title['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
    datasets['title'] = title_dataset
    models['title'] = bert.get_bert(pretrained=args.title, fp16=args.fp16).to(args.device)
    models['title'] = models['title'].eval()
    print(models['title'].config)
    
    ts = datetime.now()
    movie_companies = {'filepath': f'{args.path}/IMDB/data/movie_companies.csv',
             'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
             'columns': ['company_type_id', 'company_id', 'movie_id'],
             'intervals': {},
             'table_id': 0}
    movie_companies_dataset = dataset.get_table_dataset(**movie_companies, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=128)
    delta = datetime.now() - ts
    print(f"Loaded {movie_companies['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
    datasets['movie_companies'] = movie_companies_dataset
    models['movie_companies'] = bert.get_bert(pretrained=args.movie_companies, fp16=args.fp16).to(args.device)
    models['movie_companies'] = models['movie_companies'].eval()
    print(models['movie_companies'].config)

    ts = datetime.now()
    movie_info_idx = {'filepath': f'{args.path}/IMDB/data/movie_info_idx.csv',
             'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
             'columns': ['info_type_id', 'movie_id'],
             'intervals': {},
             'table_id': 0}
    movie_info_idx_dataset = dataset.get_table_dataset(**movie_info_idx, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=128)
    delta = datetime.now() - ts
    print(f"Loaded {movie_info_idx['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
    datasets['movie_info_idx'] = movie_info_idx_dataset
    models['movie_info_idx'] = bert.get_bert(pretrained=args.movie_info_idx, fp16=args.fp16).to(args.device)
    models['movie_info_idx'] = models['movie_info_idx'].eval()
    print(models['movie_info_idx'].config)
    
    ts = datetime.now()
    movie_keyword = {'filepath': f'{args.path}/IMDB/data/movie_keyword.csv',
             'names': ['id', 'movie_id', 'keyword_id'],
             'columns': ['keyword_id', 'movie_id'],
             'intervals': {},
             'table_id': 0}
    movie_keyword_dataset = dataset.get_table_dataset(**movie_keyword, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=128)
    delta = datetime.now() - ts
    print(f"Loaded {movie_keyword['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
    datasets['movie_keyword'] = movie_keyword_dataset
    models['movie_keyword'] = bert.get_bert(pretrained=args.movie_keyword, fp16=args.fp16).to(args.device)
    models['movie_keyword'] = models['movie_keyword'].eval()
    print(models['movie_keyword'].config)
    
    ts = datetime.now()
    movie_info = {'filepath': f'{args.path}/IMDB/data/movie_info.csv',
             'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
             'columns': ['info_type_id', 'movie_id'],
             'intervals': {},
             'table_id': 0}
    movie_info_dataset = dataset.get_table_dataset(**movie_info, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=128)
    delta = datetime.now() - ts
    print(f"Loaded {movie_info['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
    datasets['movie_info'] = movie_info_dataset
    models['movie_info'] = bert.get_bert(pretrained=args.movie_info, fp16=args.fp16).to(args.device)
    models['movie_info'] = models['movie_info'].eval()
    print(models['movie_info'].config)
    
    ts = datetime.now()
    cast_info = {'filepath': f'{args.path}/IMDB/data/cast_info.csv',
             'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
             'columns': ['role_id', 'nr_order', 'movie_id'],
             'intervals': {'nr_order': [8, 64, 256]},
             'table_id': 0}
    cast_info_dataset = dataset.get_table_dataset(**cast_info, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=128)
    delta = datetime.now() - ts
    print(f"Loaded {cast_info['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
    datasets['cast_info'] = cast_info_dataset
    models['cast_info'] = bert.get_bert(pretrained=args.cast_info, fp16=args.fp16).to(args.device)
    models['cast_info'] = models['cast_info'].eval()
    print(models['cast_info'].config)
    
    print('Models and datasets are ready to use')

    with open(args.csv, 'r') as f:
        workload = pd.read_csv( f,
                                delimiter='|',
                                names=['query', 1, 'job-light', 2, 'cardinality'],
                                usecols=['query', 'job-light', 'cardinality'],
                                )[['query', 'job-light', 'cardinality']]
        workload['tables'] = 1

        # Fast-AGMS
        workload['sketch'] = 1.0
        workload['sketch_err'] = 1.0

        workload['approx_sketch'] = 1.0
        workload['approx_sketch_err'] = 1.0

        workload['upper_sketch'] = 1.0
        workload['upper_sketch_err'] = 1.0

        # Same-Sign Only Fast-AGMS
        workload['signed'] = 1.0
        workload['signed_err'] = 1.0

        workload['approx_signed'] = 1.0
        workload['approx_signed_err'] = 1.0

        workload['upper_signed'] = 1.0
        workload['upper_signed_err'] = 1.0

        # Same-Sign Unbiased Fast-AGMS
        workload['unbiased'] = 1.0
        workload['unbiased_err'] = 1.0

        workload['approx_unbiased'] = 1.0
        workload['approx_unbiased_err'] = 1.0

        workload['upper_unbiased'] = 1.0
        workload['upper_unbiased_err'] = 1.0

        # Count-Min
        workload['counts'] = 1.0
        workload['counts_err'] = 1.0

        workload['approx_counts'] = 1.0
        workload['approx_counts_err'] = 1.0

        workload['upper_counts'] = 1.0
        workload['upper_counts_err'] = 1.0

        # Count-Mean-Median
        workload['cmm'] = 1.0
        workload['cmm_err'] = 1.0

        workload['approx_cmm'] = 1.0
        workload['approx_cmm_err'] = 1.0

        workload['upper_cmm'] = 1.0
        workload['upper_cmm_err'] = 1.0

        # Independent Joint Size Product
        workload['joint'] = 1.0
        workload['joint_err'] = 1.0

        # COMPASS Count-Mean-Median
        workload['compass'] = 1.0
        workload['compass_err'] = 1.0

        workload['approx_compass'] = 1.0
        workload['approx_compass_err'] = 1.0

        workload['upper_compass'] = 1.0
        workload['upper_compass_err'] = 1.0

        # COMPASS Count-Mean-Median
        workload['ccmm'] = 1.0
        workload['ccmm_err'] = 1.0

        workload['approx_ccmm'] = 1.0
        workload['approx_ccmm_err'] = 1.0

        workload['upper_ccmm'] = 1.0
        workload['upper_ccmm_err'] = 1.0

        # latency
        workload['sketch_time'] = 0.0
        workload['model_time'] = 0.0


    for i, row in enumerate(workload.iloc()):
        query_start = datetime.now()
        query = row['query']
        print(f'{i}: {query}')

        from_end = query.find('FROM') + 4
        where_start = query.find('WHERE')

        tables_and_abbrvs = query[from_end:where_start].split(',')

        abbrv_to_table = {}
        keys = {}
        join_count = {}
        predicates = {}
        for table_abbrv in tables_and_abbrvs:
            table_name, abbrv = table_abbrv.strip().split(' ')
            table_name = table_name.strip()
            abbrv = abbrv.strip()
            abbrv_to_table[abbrv] = table_name
            keys[table_name] = None
            join_count[table_name] = 0
            predicates[table_name] = {}

        # if (len(keys)) != 3: continue;
        num_tables = len(keys)

        where_end = where_start + 5
        and_list = query[where_end:-1].split('AND')

        for p in and_list:
            if '=' in p:
                op = '='
            if '>' in p:
                p = p.replace('>=', '>', 1)
                op = '>'
            if '<' in p:
                p = p.replace('<=', '<', 1)
                op = '<'
            left, right = p.split(op)
            left = left.split('.')
            triplet = [
                    left[1].strip(),
                    '==' if op == '=' else op,
                    right.strip()
            ]
            right = right.split('.')
            left = list(map(str.strip, left))
            right = list(map(str.strip, right))
            assert len(left) == 2, f'predicate: {p} --> ({left}, {op}, {right})'
            if len(right) == 2 and op == '=' and right[0] in abbrv_to_table:
                keys[abbrv_to_table[left[0]]] = left[1]
                keys[abbrv_to_table[right[0]]] = right[1]

                # increment join counter for each table
                join_count[abbrv_to_table[left[0]]] += 1
                join_count[abbrv_to_table[right[0]]] += 1
            else:
                if triplet[2][0] == "'" and triplet[2][-1] == "'":
                    triplet[2] = triplet[2].replace("'", '')
                else:
                    triplet[2] = float(triplet[2])
                table = abbrv_to_table[left[0]]
                col = triplet[0]
                val = triplet[2]
                if table not in predicates:
                    predicates[table] = {col: {}}
                if col not in predicates[table]:
                    predicates[table][col] = {}
                predicates[table][col][op] = val 

        # determine middle table
        middle_table = None
        if num_tables > 2:
            for table_name, num_joins in join_count.items():
                if num_joins > 1:
                    if middle_table is None:
                        middle_table = (table_name, keys[table_name])
                    else:
                        print('\tERROR!!! multiple middle tables detected: ', middle_table, table_name)
                        exit()

        # remove middle table from keys
        # if middle_table and middle_table[0] in keys:
        #     keys.pop(middle_table[0])

        print('\tmiddle table ', middle_table)
        print('\tjoins ', f'{num_tables} tables')
        print('\tkeys ', keys)
        print('\tpredicates ', predicates)

        with torch.inference_mode():
            estimates = estimate_join(models, predicates, keys, datasets, attention_mask=args.mask, decompose_limit=args.decompose_limit)

        card = row['cardinality']
        # Fast-AGMS
        workload.loc[i, 'sketch'] = float(estimates['Fast-AGMS'])
        workload.loc[i, 'sketch_err'] = float(qerror(estimates['Fast-AGMS'], card))
        workload.loc[i, 'approx_sketch'] = float(estimates['Approx. Fast-AGMS'])
        workload.loc[i, 'approx_sketch_err'] = float(qerror(estimates['Approx. Fast-AGMS'], card))
        workload.loc[i, 'upper_sketch'] = float(estimates['Upper Fast-AGMS'])
        workload.loc[i, 'upper_sketch_err'] = float(qerror(estimates['Upper Fast-AGMS'], card))
        # Same-Sign
        workload.loc[i, 'signed'] = float(estimates['Same-Sign'])
        workload.loc[i, 'signed_err'] = float(qerror(estimates['Same-Sign'], card))
        workload.loc[i, 'approx_signed'] = float(estimates['Approx. Same-Sign'])
        workload.loc[i, 'approx_signed_err'] = float(qerror(estimates['Approx. Same-Sign'], card))
        workload.loc[i, 'upper_signed'] = float(estimates['Upper Same-Sign'])
        workload.loc[i, 'upper_signed_err'] = float(qerror(estimates['Upper Same-Sign'], card))
        # Same-Sign
        workload.loc[i, 'unbiased'] = float(estimates['Same-Unbiased'])
        workload.loc[i, 'unbiased_err'] = float(qerror(estimates['Same-Unbiased'], card))
        workload.loc[i, 'approx_unbiased'] = float(estimates['Approx. Same-Unbiased'])
        workload.loc[i, 'approx_unbiased_err'] = float(qerror(estimates['Approx. Same-Unbiased'], card))
        workload.loc[i, 'upper_unbiased'] = float(estimates['Upper Same-Unbiased'])
        workload.loc[i, 'upper_unbiased_err'] = float(qerror(estimates['Upper Same-Unbiased'], card))
        # Count-Min
        workload.loc[i, 'counts'] = float(estimates['Count-Min'])
        workload.loc[i, 'counts_err'] = float(qerror(estimates['Count-Min'], card))
        workload.loc[i, 'approx_counts'] = float(estimates['Approx. Count-Min'])
        workload.loc[i, 'approx_counts_err'] = float(qerror(estimates['Approx. Count-Min'], card))
        workload.loc[i, 'upper_counts'] = float(estimates['Upper Count-Min'])
        workload.loc[i, 'upper_counts_err'] = float(qerror(estimates['Upper Count-Min'], card))
        # Count-Mean-Median
        workload.loc[i, 'cmm'] = float(estimates['Count-Mean-Median'])
        workload.loc[i, 'cmm_err'] = float(qerror(estimates['Count-Mean-Median'], card))
        workload.loc[i, 'approx_cmm'] = float(estimates['Approx. Count-Mean-Median'])
        workload.loc[i, 'approx_cmm_err'] = float(qerror(estimates['Approx. Count-Mean-Median'], card))
        workload.loc[i, 'upper_cmm'] = float(estimates['Upper Count-Mean-Median'])
        workload.loc[i, 'upper_cmm_err'] = float(qerror(estimates['Upper Count-Mean-Median'], card))
        # Joint product
        workload.loc[i, 'joint'] = float(estimates['Joint Product'])
        workload.loc[i, 'joint_err'] = float(qerror(estimates['Joint Product'], card))
        # COMPASS Fast-AGMS
        workload.loc[i, 'compass'] = float(estimates['COMPASS'])
        workload.loc[i, 'compass_err'] = float(qerror(estimates['COMPASS'], card))
        workload.loc[i, 'approx_compass'] = float(estimates['Approx. COMPASS'])
        workload.loc[i, 'approx_compass_err'] = float(qerror(estimates['Approx. COMPASS'], card))
        workload.loc[i, 'upper_compass'] = float(estimates['Upper COMPASS'])
        workload.loc[i, 'upper_compass_err'] = float(qerror(estimates['Upper COMPASS'], card))
        # COMPASS Count-Mean-Median
        workload.loc[i, 'ccmm'] = float(estimates['COMPASS CMM'])
        workload.loc[i, 'ccmm_err'] = float(qerror(estimates['COMPASS CMM'], card))
        workload.loc[i, 'approx_ccmm'] = float(estimates['Approx. COMPASS CMM'])
        workload.loc[i, 'approx_ccmm_err'] = float(qerror(estimates['Approx. COMPASS CMM'], card))
        workload.loc[i, 'upper_ccmm'] = float(estimates['Upper COMPASS CMM'])
        workload.loc[i, 'upper_ccmm_err'] = float(qerror(estimates['Upper COMPASS CMM'], card))

        workload.loc[i, 'tables'] = num_tables

        workload.loc[i, 'sketch_time'] = estimates['Sketch Time']
        workload.loc[i, 'model_time'] = estimates['Model Time']

        print(workload.loc[i])

        delta = datetime.now() - query_start
        print(f'Subquery {i} finished in {delta}s')
        print(flush=True)
        # break # testing

    workload.to_csv(args.writefile, index=False)

    times = [c for c in workload.columns if '_time' in c]
    print(workload[times].describe().transpose().sort_values('mean').to_string(float_format="{:,.2f}".format))
    print()
    
    delta = datetime.now() - ts
    print(f'Total program runtime: {delta}s')
    err = [c for c in workload.columns if '_err' in c]
    errors = workload[err].describe().transpose().sort_values('mean').to_string(float_format="{:,.2f}".format)
    print(errors)
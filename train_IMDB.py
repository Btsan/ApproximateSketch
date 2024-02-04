import argparse
from datetime import datetime, timedelta

from numpy.random import RandomState
from torch.utils.data import ConcatDataset
from transformers import Trainer, TrainingArguments, TrainerCallback

import xi
import bert
import dataset

class TimingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.ts = datetime.now()
    
    def on_log(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            now = datetime.now()
            elapsed = now - self.ts
            steps_per_second = state.global_step / elapsed.total_seconds()
            remaining = (elapsed / state.global_step) * (state.max_steps - state.global_step)
            print(f'{state.global_step/state.max_steps:.0%} {state.global_step:>15}/{state.max_steps:<15} [{elapsed} < {remaining}, \t{steps_per_second:>15.2f} it/s]')
        return super().on_log(args, state, control, **kwargs)
        

if __name__ == '__main__':
    # DO NOT CHANGE THIS SEED
    rs = RandomState(2**31-1)
    # The hash functions used to hash the training data depends on this seed

    parser = argparse.ArgumentParser(description='trains a bert model on IMDB table(s)')
    parser.add_argument('--num_hash', default=5, type=int, help='number of independent hash functions')
    parser.add_argument('--num_bins', default=64, type=int, help='number of buckets per hash function')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size per device')
    parser.add_argument('--batch_mult', default=8, type=int, help='gradient accumulation steps i.e., batch_size * batch_mult')
    parser.add_argument('--epochs', default=1, type=int, help='num training epochs')
    parser.add_argument('--tqdm', action='store_false', help='enable tqdm progress display printing')
    parser.add_argument('--dataloaders', default=8, type=int, help='num dataloader workers')
    parser.add_argument('--resume', action='store_true', help='enable to resume from last training checkpoint')
    parser.add_argument('--pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('--save', default=None, type=str, help='model save directory')
    parser.add_argument('--path', default='./', type=str, help='path to IMDB data directory containing all table CSV files')
    parser.add_argument('--all', action='store_true', help='trains on all tables otherwise use specified tables')
    parser.add_argument('--title', action='store_true', help='trains on title')
    parser.add_argument('--movie_companies', action='store_true', help='trains on movie_companies')
    parser.add_argument('--movie_info_idx', action='store_true', help='trains on movie_info_idx')
    parser.add_argument('--movie_keyword', action='store_true', help='trains on movie_keyword')
    parser.add_argument('--movie_info', action='store_true', help='trains on movie_info')
    parser.add_argument('--cast_info', action='store_true', help='trains on cast_info')
    parser.add_argument('--pad_length', default=None, type=int, help='length to pad model input sequence (`None` auto determine)')
    parser.add_argument('--max_length', default=256, type=int, help='maximum allowable model input sequence length (128 suffices for JOB-light)')
    args = parser.parse_args()
    print(args)
    assert any([args.all, args.title, args.movie_companies, args.movie_info_idx, args.movie_keyword, args.movie_info, args.cast_info]), 'No IMDB tables specified for training'

    model_type = 'BERT'
    table_names = []
    if args.all:
        table_names.append('all')
    else:
        if args.title:
            table_names.append('title')
        if args.movie_companies:
            table_names.append('movie_companies')
        if args.movie_info_idx:
            table_names.append('movie_info_idx')
        if args.movie_keyword:
            table_names.append('movie_keyword')
        if args.movie_info:
            table_names.append('movie_info')
        if args.cast_info:
            table_names.append('cast_info')

    training_args = TrainingArguments(f"./{model_type}__{args.num_hash}x{args.num_bins}__{'__'.join(table_names)}__checkpoints",
                                      overwrite_output_dir=args.resume,
                                      per_device_train_batch_size=args.batch_size,
                                      gradient_accumulation_steps=args.batch_mult,
                                      num_train_epochs=args.epochs,
                                      log_level='info',
                                      logging_dir=f"./{model_type}__{args.num_hash}x{args.num_bins}__{'__'.join(table_names)}__logs",
                                      logging_steps=100,
                                      save_strategy='steps',
                                      save_steps=500,
                                      disable_tqdm=args.tqdm,
                                      dataloader_num_workers=args.dataloaders,
                                      resume_from_checkpoint=args.resume,
                                      )
    
    # hash functions
    bin_hashes = [xi.B_Xi(args.num_bins, rs.randint(2**32), rs.randint(2**32)) for _ in range(args.num_hash)]
    one_hashes = [xi.Xi(rs.randint(2**32), rs.randint(2**32)) for _ in range(args.num_hash)]


    datasets = []

    if args.all or args.title:
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
                'table_id': len(datasets)}
        title_dataset = dataset.get_table_dataset(**title, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=args.pad_length)
        print(title_dataset.table.describe().to_string(float_format='{:,.2f}'.format))
        title_dataset.stats()
        delta = datetime.now() - ts
        print(f"Loaded {title['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
        datasets.append(title_dataset)
    
    if args.all or args.movie_companies:
        ts = datetime.now()
        movie_companies = {'filepath': f'{args.path}/IMDB/data/movie_companies.csv',
                'names': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                'columns': ['company_type_id', 'company_id', 'movie_id'],
                'intervals': {},
                'table_id': len(datasets)}
        movie_companies_dataset = dataset.get_table_dataset(**movie_companies, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=args.pad_length)
        print(movie_companies_dataset.table.describe().to_string(float_format='{:,.2f}'.format))
        movie_companies_dataset.stats()
        delta = datetime.now() - ts
        print(f"Loaded {movie_companies['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
        datasets.append(movie_companies_dataset)

    if args.all or args.movie_info_idx:
        ts = datetime.now()
        movie_info_idx = {'filepath': f'{args.path}/IMDB/data/movie_info_idx.csv',
                'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                'columns': ['info_type_id', 'movie_id'],
                'intervals': {},
                'table_id': len(datasets)}
        movie_info_idx_dataset = dataset.get_table_dataset(**movie_info_idx, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=args.pad_length)
        print(movie_info_idx_dataset.table.describe().to_string(float_format='{:,.2f}'.format))
        movie_info_idx_dataset.stats()
        delta = datetime.now() - ts
        print(f"Loaded {movie_info_idx['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
        datasets.append(movie_info_idx_dataset)
    
    if args.all or args.movie_keyword:
        ts = datetime.now()
        movie_keyword = {'filepath': f'{args.path}/IMDB/data/movie_keyword.csv',
                'names': ['id', 'movie_id', 'keyword_id'],
                'columns': ['keyword_id', 'movie_id'],
                'intervals': {},
                'table_id': len(datasets)}
        movie_keyword_dataset = dataset.get_table_dataset(**movie_keyword, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=args.pad_length)
        print(movie_keyword_dataset.table.describe().to_string(float_format='{:,.2f}'.format))
        movie_keyword_dataset.stats()
        delta = datetime.now() - ts
        print(f"Loaded {movie_keyword['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
        datasets.append(movie_keyword_dataset)
        
    if args.all or args.movie_info:
        ts = datetime.now()
        movie_info = {'filepath': f'{args.path}/IMDB/data/movie_info.csv',
                'names': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                'columns': ['info_type_id', 'movie_id'],
                'intervals': {},
                'table_id': len(datasets)}
        movie_info_dataset = dataset.get_table_dataset(**movie_info, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=args.pad_length)
        print(movie_info_dataset.table.describe().to_string(float_format='{:,.2f}'.format))
        movie_info_dataset.stats()
        delta = datetime.now() - ts
        print(f"Loaded {movie_info['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
        datasets.append(movie_info_dataset)
    
    if args.all or args.cast_info:
        ts = datetime.now()
        cast_info = {'filepath': f'{args.path}/IMDB/data/cast_info.csv',
                'names': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                'columns': ['role_id', 'nr_order', 'movie_id'],
                'intervals': {'nr_order': [8, 64, 256]},
                'table_id': len(datasets)}
        cast_info_dataset = dataset.get_table_dataset(**cast_info, bin_hashes=bin_hashes, one_hashes=one_hashes, max_length=args.pad_length)
        print(cast_info_dataset.table.describe().to_string(float_format='{:,.2f}'.format))
        cast_info_dataset.stats()
        delta = datetime.now() - ts
        print(f"Loaded {cast_info['filepath']:<50} {delta.total_seconds():>25,.2f}s ({delta})")
        datasets.append(cast_info_dataset)
    
    model = bert.get_bert(num_tables=len(datasets),
                          num_hashes=args.num_hash,
                          num_bins=args.num_bins,
                          num_heads=4,
                          num_layers=6,
                          embed_size=64,
                          feedforward_size=256,
                          max_length=args.max_length,
                          pretrained=args.pretrained,)
    print(model.config)
    
    # begin training
    train_dataset = ConcatDataset(datasets)
    model.train(True)
    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=train_dataset,)
    trainer.add_callback(TimingCallback)
    trainer.train()

    if args.save:
        model.save_pretrained(save_directory=f"./{model_type}__{args.num_hash}x{args.num_bins}__{'__'.join(table_names)}__{args.epochs}__{args.save}")

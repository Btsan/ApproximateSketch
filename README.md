<meta name="robots" content="noindex">

# ApproximateSketch

Approximate the sketch of any arbitrary selection.

### Requirements

- Python 3.6+
- [transformers](https://huggingface.co/docs/transformers) 4.26+
- [pytorch](https://pytorch.org/get-started/locally/) 1.13+
- pandas 1.5+
- numpy 1.23+

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c huggingface transformers
conda install pandas numpy
```
Restarting your shell may be necessary to use `transformers` after installing.

### Data

The IMDB snapshot used for training can be downloaded from [`http://homepages.cwi.nl/~boncz/job/imdb.tgz`](http://homepages.cwi.nl/~boncz/job/imdb.tgz).
Extract the CSV files into the same directory, whose path `"$PATH_TO_CSV"/IMDB/data/` can be provided to the training scripts.

### Training

Pretrained models for approximating 5x64 and 5x512 Fast-AGMS sketches (doubles as 5x128 and 5x1024 Count-Min) are available in folders [BERT_5x64/](./BERT_5x64) and [BERT_5x512](./BERT_5x512).

Models for approximate 5x4096 Fast-AGMS (5x8192 Count-Min) could not be uploaded, but can be trained with the following:

```bash
python train_IMDB.py --num_hash 5 --num_bins 4096 --batch_size 128 --epochs 1 --save BERT_5x4096/ --path "$PATH_TO_CSV" --title
python train_IMDB.py --num_hash 5 --num_bins 4096 --batch_size 128 --epochs 1 --save BERT_5x4096/ --path "$PATH_TO_CSV" --movie_companies
python train_IMDB.py --num_hash 5 --num_bins 4096 --batch_size 128 --epochs 1 --save BERT_5x4096/ --path "$PATH_TO_CSV" --movie_info_idx
python train_IMDB.py --num_hash 5 --num_bins 4096 --batch_size 128 --epochs 1 --save BERT_5x4096/ --path "$PATH_TO_CSV" --movie_keyword
python train_IMDB.py --num_hash 5 --num_bins 4096 --batch_size 128 --epochs 1 --save BERT_5x4096/ --path "$PATH_TO_CSV" --movie_info
python train_IMDB.py --num_hash 5 --num_bins 4096 --batch_size 128 --epochs 1 --save BERT_5x4096/ --path "$PATH_TO_CSV" --cast_info
```

The hash functions used for training of our approximate sketch models are in [`xi.py`](./xi.py)

### Evaluation

Cardinality estimation can be evaluated for the workloads [JOB-light](./job_light_sub_query_with_star_join.sql.txt) and [JOB-light-ranges](./job_light_ranges_subqueries.sql.txt).

Example of evaluating 5x64 sketches on job-light:
```bash
python test_IMDB.py --num_hash 5 \
  --num_bins 64 \
  --csv job_light_sub_query_with_star_join.sql.txt \
  --title BERT_5x64/title__10c/ \
  --movie_companies BERT_5x64/movie_companies__10c/ \
  --movie_info_idx BERT_5x64/movie_info_idx__10c/ \
  --movie_keyword BERT_5x64/movie_keyword__10c/ \
  --cast_info BERT_5x64/cast_info__10c/ \
  --path "$PATH_TO_CSV" \
  --writefile JOB_light5x64_decompose16.csv \
  --decompose_limit 16
```
Additional examples in [`evaluate.sh`](./evaluate.sh).

For query performance, refer to the modified PostgreSQL installation in [another work that we have cited](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark).

### Results
After running the evaluation script, you should have an output similar to [this log file](./JOB_light_5x4096_decompose16.log).
The table of Q-errors printed at the end lists many different methods, mentioned by abbreviations, which we now explain:

| Abbreviation | Explanation |
| --- | --- |
| ccmm | Count-Min-Mean sketch with COMPASS merge for multiway join size estimation |
| cmm | Count-Min-Mean sketch with a novel sign-based estimation process for multiway joins|
| compass | Fast-AGMS sketch with COMPASS merge for multiway join size estimation |
| signed | Fast-AGMS sketch with a novel sign-based estimation process for multiway joins|
| counts | Count-Min sketch with its classic upper-bound estimate|
| sketch | Fast-AGMS sketch, simply elementwise multiplied for multiway joins|
| unbiased | Fast-AGMS sketch with another novel sign-based estimation process for multiway joins|
| joint | cartesian product of selections as join size estimation|

Methods may be affixed with `upper_` or `approx_` to refer to their approximated sketches.
Novel estimation methods listed above but not mentioned in our paper simply did not perform well in our evaluation.

### Citation

TBA

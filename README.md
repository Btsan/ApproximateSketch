# ApproximateSketch

Approximate the sketch of any arbitrary selection. ([paper](https://doi.org/10.1145/3639321))

### Requirements

- Python 3.6+
- [transformers](https://huggingface.co/docs/transformers) 4.26+
- [pytorch](https://pytorch.org/get-started/locally/) 1.13+
- [pandas](https://pandas.pydata.org/docs/getting_started/) 1.5+
- [numpy](https://numpy.org/install) 1.23+

``` bash
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

Models for approximate 5x4096 Fast-AGMS (5x8192 Count-Min) can be trained with the following:

``` bash
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
``` bash
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

For query performance, refer to the modified PostgreSQL installation from [another work that we have cited](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark).

### Results
After running the evaluation script, you should have an output similar to [this log file](./JOB_light_5x4096_decompose16.log).
The table of Q-errors printed at the end lists many different methods, mentioned by abbreviations, which we now explain:

| Abbreviation | Explanation |
| --- | --- |
| ccmm | Count-Mean-Min sketch with COMPASS merge for multiway join size estimation |
| cmm | Count-Mean-Min sketch with a novel sign-based estimation process for multiway joins|
| compass | Fast-AGMS sketch with COMPASS merge for multiway join size estimation |
| signed | Fast-AGMS sketch with a novel sign-based estimation process for multiway joins|
| counts | Count-Min sketch with its classic upper-bound estimate|
| sketch | Fast-AGMS sketch, simply elementwise multiplied for multiway joins|
| unbiased | Fast-AGMS sketch with another novel sign-based estimation process for multiway joins|
| joint | cartesian product of selections as join size estimation|

Methods may be affixed with `upper_` or `approx_` to refer to their approximated sketches.

### Citation

``` bibtex
@article{10.1145/3639321,
author = {Tsan, Brian and Datta, Asoke and Izenov, Yesdaulet and Rusu, Florin},
title = {Approximate Sketches},
year = {2024},
issue_date = {February 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {2},
number = {1},
url = {https://doi.org/10.1145/3639321},
doi = {10.1145/3639321},
abstract = {Sketches are single-pass small-space data summaries that can quickly estimate the cardinality of join queries. However, sketches are not directly applicable to join queries with dynamic filter conditions --- where arbitrary selection predicate(s) are applied --- since a sketch is limited to a fixed selection. While multiple sketches for various selections can be used in combination, they each incur individual storage and maintenance costs. Alternatively, exact sketches can be built during runtime for every selection. To make this process scale, a high-degree of parallelism --- available in hardware accelerators such as GPUs --- is required. Therefore, sketch usage for cardinality estimation in query optimization is limited. Following recent work that applies transformers to cardinality estimation, we design a novel learning-based method to approximate the sketch of any arbitrary selection, enabling sketches for join queries with filter conditions. We train a transformer on each table to estimate the sketch of any subset of the table, i.e., any arbitrary selection. Transformers achieve this by learning the joint distribution amongst table attributes, which is equivalent to a multidimensional sketch. Subsequently, transformers can approximate any sketch, enabling sketches for join cardinality estimation. In turn, estimating joins via approximate sketches allows tables to be modeled individually and thus scales linearly with the number of tables. We evaluate the accuracy and efficacy of approximate sketches on queries with selection predicates consisting of conjunctions of point and range conditions. Approximate sketches achieve similar accuracy to exact sketches with at least one order of magnitude less overhead.},
journal = {Proc. ACM Manag. Data},
month = {mar},
articleno = {66},
numpages = {24},
keywords = {cardinality estimation, database sketch, neural networks, synopsis}
}
```

#!/bin/bash 

bins=$1
decompose=$2

ls BERT_5x"$bins"/*

# job_light_sub_query_with_star_join
nohup python test_IMDB.py \
	--num_bins $bins \
	--csv job_light_sub_query_with_star_join.sql.txt \
	--title BERT_5x"$bins"/title__10c/ \
	--movie_companies BERT_5x"$bins"/movie_companies__10c/ \
	--movie_info_idx BERT_5x"$bins"/movie_info_idx__10c/ \
	--movie_keyword BERT_5x"$bins"/movie_keyword__10c/ \
	--movie_info BERT_5x"$bins"/movie_info__10c/ \
	--cast_info BERT_5x"$bins"/cast_info__10c/ \
	--writefile JOB_light_5x"$bins"_decompose"$decompose".csv \
	--decompose_limit $decompose \
	&> JOB_light_5x"$bins"_decompose"$decompose".log <&- &

# job_light_ranges_subqueries
nohup python test_IMDB.py \
	--num_bins $bins \
	--csv job_light_ranges_subqueries.sql.txt \
	--title BERT_5x"$bins"/title__10c/ \
	--movie_companies BERT_5x"$bins"/movie_companies__10c/ \
	--movie_info_idx BERT_5x"$bins"/movie_info_idx__10c/ \
	--movie_keyword BERT_5x"$bins"/movie_keyword__10c/ \
	--movie_info BERT_5x"$bins"/movie_info__10c/ \
	--cast_info BERT_5x"$bins"/cast_info__10c/ \
	--writefile JOB_light_ranges_5x"$bins"_decompose"$decompose".csv \
	--decompose_limit $decompose \
	&> JOB_light_ranges_5x"$bins"_decompose"$decompose".log <&- &
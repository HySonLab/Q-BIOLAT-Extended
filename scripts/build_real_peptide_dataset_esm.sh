#!/bin/bash

seed=42

for num_samples in 1000 2000 5000 10000; do
	echo "Sample ${num_samples} proteins from GFP"

	python examples/make_subset_csv.py \
  		--input-csv data/proteingym/gfp.csv \
  		--output-csv data/proteingym/gfp_${num_samples}.csv \
  		--n $num_samples \
  		--seed $seed

	for dim in 8 16 32 64; do
		echo "Build ${dim}-dim binarized ESM embedding for ${num_samples} samples of GFP"

		python examples/build_real_peptide_dataset_esm.py \
			--input-csv data/proteingym/gfp_${num_samples}.csv \
			--output-npz artifacts/gfp_${num_samples}_esm_binary_${dim}.npz \
  			--project-dim $dim \
  			--binarize median \
  			--device cpu
	done
done


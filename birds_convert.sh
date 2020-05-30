python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=cu_birds \
  --cu_birds_data_root=db/CUB_200_2011 \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

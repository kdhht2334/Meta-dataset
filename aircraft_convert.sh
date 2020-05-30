python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
  --dataset=aircraft \
  --aircraft_data_root=db/fgvc-aircraft-2013b \
  --splits_root=$SPLITS \
  --records_root=$RECORDS

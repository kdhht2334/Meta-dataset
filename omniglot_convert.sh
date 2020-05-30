python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
      --dataset=omniglot \
      --omniglot_data_root=db/omniglot \
      --splits_root=$SPLITS \
      --records_root=$RECORDS

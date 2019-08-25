#!/usr/bin/env bash


export BERT_BASE_DIR=/data/users/yuanhaoda/Bert-n-Pals/bert_models/uncased_L-12_H-768_A-12
export BERT_PYTORCH_DIR=/data/users/yuanhaoda/Bert-n-Pals/bert_models/uncased_L-12_H-768_A-12
export GLUE_DIR=/data/users/yuanhaoda/Bert-n-Pals/bert_models/glue/glue_data
export SAVE_DIR=/data/users/yuanhaoda/Bert-n-Pals/bert_models/tmp/saved

python run_multi_task.py \
  --seed 42 \
  --output_dir $SAVE_DIR/pals \
  --tasks all \
  --sample 'anneal'\
  --multi \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/pals_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 25.0 \
  --gradient_accumulation_steps 1
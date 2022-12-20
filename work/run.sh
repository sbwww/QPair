RDROP=0
LR=4e-5
TRAIN_BATCH=64
EPOCHS=4

# python train.py --model_name_or_path ernie-gram-zh --train_set ./data/train_merge.tsv --dev_set ./data/dev_merge.tsv --device gpu --eval_step 1000 --output_dir ./model/ernie-gram-rdrop_${RDROP} --train_batch_size ${TRAIN_BATCH} --num_train_epochs ${EPOCHS} --learning_rate ${LR} --rdrop_coef ${RDROP}

# python predict.py --model_name_or_path ./model/ernie-gram-rdrop_${RDROP} --input_file ./data/test_merge.tsv --result_file ./model/ernie-gram-rdrop_${RDROP}/prediction.csv --device gpu --batch_size 32

python train.py --model_name_or_path ernie-3.0-xbase-zh --train_set ./data/train_merge.tsv --dev_set ./data/dev_merge.tsv --device gpu --eval_step 1000 --output_dir ./model/ernie-3-xbase --train_batch_size ${TRAIN_BATCH} --num_train_epochs ${EPOCHS} --learning_rate ${LR} --rdrop_coef ${RDROP}

python predict.py --model_name_or_path ./model/ernie-3-xbase --input_file ./data/test_merge.tsv --result_file ./model/ernie-3-xbase/prediction.csv --device gpu --batch_size 64

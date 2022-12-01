RDROP=0
LR=2e-5
EPOCHS=3

python -u -m paddle.distributed.launch --gpus "0" train.py \
    --model_name_or_path ernie-gram-zh \
    --train_set ../data/train_merge.tsv \
    --dev_set ../data/dev_merge.tsv \
    --device gpu \
    --eval_step 3000 \
    --save_dir ../model/ernie-gram-rdrop_${RDROP} \
    --train_batch_size 32 \
    --epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --rdrop_coef ${RDROP}

python -u -m paddle.distributed.launch --gpus "0" predict.py \
    --model_name_or_path ../model/ernie-gram-rdrop_${RDROP}/model_state.pdparams \
    --input_file ../model/test.tsv \
    --result_file ../model/ernie-gram-rdrop_${RDROP}/result.txt \
    --device gpu \
    --save_dir ../model/ernie-gram-rdrop_${RDROP} \
    --batch_size 32

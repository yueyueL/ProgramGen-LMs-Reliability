#################
##Salesforce/codet5-base  microsoft/codereviewer  Salesforce/codet5p-220m  razent/cotext-1-ccg  SEBIS/code_trans_t5_base_code_comment_generation_java
datasize=medium
tokenizer_name=Salesforce/codet5-base
model_name=Salesforce/codet5-base
output_dir=Bugs2Fix/output/$datasize/codet5

##############################################################################################
##############################################################################################

#####train

CUDA_LAUNCH_BLOCKING=0 python3 Bugs2Fix/model.py \
    --model_name=model.bin \
    --output_dir=$output_dir \
    --tokenizer_name=$tokenizer_name \
    --model_name_or_path=$model_name \
    --do_train \
    --train_data_file=Bugs2Fix/data/$datasize/train \
    --eval_data_file=Bugs2Fix/data/$datasize/valid \
    --test_data_file=Bugs2Fix/data/$datasize/test \
    --epochs 15 \
    --encoder_block_size 512 \
    --decoder_block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 


#####test
CUDA_LAUNCH_BLOCKING=0 python3 Bugs2Fix/model.py \
    --output_dir=$output_dir \
    --model_name=model.bin \
    --tokenizer_name=$tokenizer_name \
    --model_name_or_path=$model_name \
    --do_test \
    --test_data_file=Bugs2Fix/data/$datasize/test \
    --encoder_block_size 512 \
    --decoder_block_size 512 \
    --eval_batch_size 1 \
    --num_beams 1

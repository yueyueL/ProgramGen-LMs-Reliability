datasize=small
output_dir=/path/to/output/$datasize/codegpt
data_dir=/path/to/data/$datasize/


python3 run.py \
        --do_train \
        --do_eval \
        --model_type gpt2 \
        --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
        --train_filename $data_dir/train.buggy-fixed.buggy,$data_dir/train.buggy-fixed.fixed \
        --dev_filename $data_dir/valid.buggy-fixed.buggy,$data_dir/valid.buggy-fixed.fixed \
        --test_filename $data_dir/test.buggy-fixed.buggy,$data_dir/test.buggy-fixed.fixed \
        --output_dir $output_dir \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 1 \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --learning_rate 2e-5 \
        --num_train_epochs 15 \
        2>&1 


python3 run.py \
        --do_test \
        --model_type gpt2 \
        --model_name_or_path microsoft/CodeGPT-small-java-adaptedGPT2 \
        --load_model_path $output_dir/checkpoint-best-ppl/pytorch_model.bin \
        --train_filename $data_dir/train.buggy-fixed.buggy,$data_dir/train.buggy-fixed.fixed \
        --dev_filename $data_dir/valid.buggy-fixed.buggy,$data_dir/valid.buggy-fixed.fixed \
        --test_filename $data_dir/test.buggy-fixed.buggy,$data_dir/test.buggy-fixed.fixed \
        --output_dir $output_dir \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 1 \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --learning_rate 2e-5 \
        --num_train_epochs 15


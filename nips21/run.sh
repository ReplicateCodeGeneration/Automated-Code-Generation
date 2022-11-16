#################
##Salesforce/codet5-base  microsoft/codereviewer  razent/cotext-1-ccg  SEBIS/code_trans_t5_base_code_comment_generation_java
tokenizer_name=microsoft/codereviewer
model_name=microsoft/codereviewer
output_dir=nips21/data/output/java_cs/codereviewer

##############################################################################################
##############################################################################################

CUDA_LAUNCH_BLOCKING=0 python3 nips21/model.py \
    --model_name=model.bin \
    --output_dir=$output_dir \
    --tokenizer_name=$tokenizer_name \
    --model_name_or_path=$model_name \
    --do_train \
    --train_data_file=nips21/data/train \
    --eval_data_file=nips21/data/valid \
    --test_data_file=nips21/data/test \
    --epochs 15 \
    --encoder_block_size 512 \
    --decoder_block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 

CUDA_LAUNCH_BLOCKING=0 python3 nips21/model.py \
    --output_dir=$output_dir \
    --model_name=model.bin \
    --tokenizer_name=$tokenizer_name \
    --model_name_or_path=$model_name \
    --do_test \
    --test_data_file=nips21/data/test \
    --encoder_block_size 512 \
    --decoder_block_size 512 \
    --eval_batch_size 1 \
    --num_beams 1

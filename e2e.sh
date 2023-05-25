export source='cnn'
export MODEL_DIR='../models/'

#### Prepare Data

for split in valid test train;
do python extractor.py --source $source --split $split --save_name desc_data
done

for split in valid test train;
do python mask_generator.py --source $source --split $split --input_name desc_data --save_name masked_data
done

for split in valid test train;
do python alternative_desc.py --source $source --split $split --input_name masked_data --save_name desc_mcqa
done

for split in valid test train;
do python alternative_claim.py --source $source --split $split --input_name masked_data --save_name claim_mcqa
done


### Train and Evaluate

for model_name in 'T5-small' 'T5-base' 'BART';
do python generation.py --max_sent 5 --checkpoint_dir $MODEL_DIR/descgen_${model_name}_clipped --model_name $model_name --do_train --do_eval --eval_classes --eval_log logs/gen_${model_name}.log
done


for model_name in 'BERT' 'RoBERTa' 'Electra'
do python mcqa.py --max_sent 5 --total_sent 10 --checkpoint_dir $MODEL_DIR/claim_${model_name}_clipped --log_dir ${MODEL_DIR}/runs/${model_name} --model_name $model_name --task claim --do_train --do_eval --eval_classes --eval_log logs/claim_${model_name}.log
done


for model_name in 'BERT' 'RoBERTa' 'Electra'
do python mcqa.py --max_sent 5 --total_sent 10 --checkpoint_dir $MODEL_DIR/desc_${model_name}_clipped --log_dir ${MODEL_DIR}/runs/${model_name} --model_name $model_name --task desc --do_train --do_eval --eval_classes --eval_log logs/desc_${model_name}.log
done

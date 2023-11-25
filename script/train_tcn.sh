
#!/bin/bash

# Set the path to the Python module
PYTHON_MODULE="trainer.tcn_trainer"

# Set the path to the output directory
OUTPUT_DIR="results"

# Set other optional arguments
OVERWRITE_OUTPUT_DIR="true"
DO_TRAIN="true"
DO_EVAL="true"
DO_PREDICT="true"
EVALUATION_STRATEGY="steps"
PREDICTION_LOSS_ONLY="true"
PER_DEVICE_EVAL_BATCH_SIZE="8"
PER_GPU_TRAIN_BATCH_SIZE="16"
PER_GPU_EVAL_BATCH_SIZE="8"
GRADIENT_ACCUMULATION_STEPS="1"
EVAL_ACCUMULATION_STEPS="1"
EVAL_DELAY="0"
LEARNING_RATE="0.001"
WEIGHT_DECAY="0.01"
ADAM_BETA1="0.9"
ADAM_BETA2="0.999"
ADAM_EPSILON="1e-8"
MAX_GRAD_NORM="1.0"
NUM_TRAIN_EPOCHS="10"
MAX_STEPS="100000"
LR_SCHEDULER_TYPE="linear"
PER_DEVICE_TRAIN_BATCH_SIZE="16"
PREDICT_SIZE="15"
OUTPUT_SIZE="15"
EVAL_STEPS="1000"
# Run the Python module with the provided arguments
python3 -m  $PYTHON_MODULE \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir $OVERWRITE_OUTPUT_DIR \
  --do_train $DO_TRAIN \
  --do_eval $DO_EVAL \
  --do_predict $DO_PREDICT \
  --evaluation_strategy $EVALUATION_STRATEGY \
  --prediction_loss_only $PREDICTION_LOSS_ONLY \
  --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
  --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
  --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --eval_accumulation_steps $EVAL_ACCUMULATION_STEPS \
  --eval_steps $EVAL_STEPS \
  --eval_delay $EVAL_DELAY \
  --learning_rate $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --adam_epsilon $ADAM_EPSILON \
  --max_grad_norm $MAX_GRAD_NORM \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --max_steps $MAX_STEPS \
  --lr_scheduler_type $LR_SCHEDULER_TYPE \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE\
  --dataset "NASDAQ_3y" \
  --predict_size $PREDICT_SIZE\
  --output_size $OUTPUT_SIZE\

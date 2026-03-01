#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
use_prompts=1
num_prompts=100
plen=10
train=1
exp_name="demo_all_coco"
split_point=0

repo_name="/data/zyt/PDP/deformable-detr"
tr_dir="/data/zyt/coco/train2017"
val_dir="/data/zyt/coco/val2017"
task_ann_dir="/data/zyt/40_20_20/"${split_point}

freeze='backbone,encoder,decoder'
new_params='class_embed,prompts'

EXP_DIR=/data/zyt/md-detr_prototypes/run/${exp_name}

if [[ $train -gt 0 ]]
then
echo "Training ... "${exp_name}
LD_DIR=/data/zyt/PDP/run/demo_all_coco
python main.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} --repo_name=${repo_name} \
    --n_gpus 2 --batch_size 2 --epochs 8 --lr 1e-4 --lr_old 1e-5 --n_classes=81 --num_workers=4 --split_point=$split_point \
    --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len=$plen --freeze=${freeze} --viz --new_params=${new_params} \
    --start_task=2 --n_tasks=3 --save_epochs=1 --eval_epochs=1  --bg_thres=0.65 --bg_thres_topk=5 --local_query=1 --lambda_query=0.1 \
    --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint_base 'checkpoint07.pth' --checkpoint_next 'checkpoint07.pth' --resume=0
else
echo "Evaluating ..."
exp_name="outputs"
EXP_DIR=/data/zyt/PDP/run/${exp_name}
LD_DIR=/data/zyt/PDP/run/demo_all_coco

python main.py \
    --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
    --n_gpus 2 --batch_size 2 --epochs 16 --lr 1e-4 --lr_old 1e-5 --save_epochs=5 --eval_epochs=2 --n_classes=81 --num_workers=4 \
    --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len=$plen --freeze=${freeze} --new_params=${new_params} \
    --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint_base 'checkpoint07.pth' --checkpoint_next 'checkpoint07.pth' --eval --viz \
    --start_task=2 --n_tasks=3 --local_query=1
fi

# #### Resume from a checkpoint for given task and then train
# python main.py \
#     --output_dir ${EXP_DIR} --train_img_dir ${tr_dir} --test_img_dir ${val_dir} --task_ann_dir ${task_ann_dir} \
#     --n_gpus 8 --batch_size 1 --epochs 26 --lr 1e-4 --lr_old 1e-5 --n_tasks=4 \
#     --use_prompts $use_prompts --num_prompts $num_prompts --prompt_len $plen --save_epochs=5 --eval_epochs=2 \
#     --freeze=${freeze} --viz --new_params=${new_params} --n_classes=81 --num_workers=2 \
#     --start_task=2 --checkpoint_dir ${LD_DIR}'/Task_1' --checkpoint 'checkpoint10.pth' --resume=1
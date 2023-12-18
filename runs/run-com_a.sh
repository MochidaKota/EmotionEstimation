RUN_NUM=8
target_emo="comfort"
gpu_id="0"
labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize5.csv"
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize5.csv"
####################################################################################################

use_feat_list="1 0 0"
hidden_dims="2048 512"
attpool_hidden_dims="2048 512"
dropout=0.1
val_phase="True"
window_size=30
n_epochs=20


run_name="2-1_c_a_mean_ws30-ss5-adam"
pool_type="mean"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="2-1_c_a_max_ws30-ss5-adam"
pool_type="max"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="2-1_c_a_att_ws30-ss5-adam"
pool_type="att"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="2-1_c_a_mean_ws30-ss5-adamw"
pool_type="mean"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \
    --optimizer "AdamW" \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="2-1_c_a_max_ws30-ss5-adamw"
pool_type="max"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \
    --optimizer "AdamW" \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="2-1_c_a_att_ws30-ss5-adamw"
pool_type="att"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \
    --optimizer "AdamW" \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="2-1_c_a_mean_ws30-ss5-lr0.0001"
pool_type="mean"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \
    --lr 0.0001 \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="2-1_c_a_max_ws30-ss5-lr0.0001"
pool_type="max"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \
    --lr 0.0001 \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="2-1_c_a_att_ws30-ss5-lr0.0001"
pool_type="att"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --hidden_dims $hidden_dims \
    --attpool_hidden_dims $attpool_hidden_dims \
    --dropout $dropout \
    --val_phase $val_phase \
    --window_size $window_size \
    --pool_type $pool_type \
    --lr 0.0001 \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --hidden_dims $hidden_dims \
        --attpool_hidden_dims $attpool_hidden_dims \
        --dropout $dropout \
        --window_size $window_size \
        --pool_type $pool_type \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/culc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

python ../src/utils/notify-slack.py \
--message "run-com_a finish!" \
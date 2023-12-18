RUN_NUM=6
target_emo="discomfort"
gpu_id="1"
labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_nomixed_seq_labels_wsize30-ssize5.csv"
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_nomixed_seq_video_name_list_wsize30-ssize5.csv"
####################################################################################################
use_feat_list="0 0 1"
batchnorm="True"
maxpool="False"
val_phase="True"
window_size=30
n_epochs=20

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl"
run_name="2-2_d_h_womp-logits-nch2.2_ws30-ss5-adamw"
hid_channels="8 512"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.2-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --optimizer "AdamW" \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --hp_feats_path $hp_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \
        --window_size $window_size \

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

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl"
run_name="2-2_d_h_womp-logits-nch2.3_ws30-ss5-adamw"
hid_channels="32 512"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.2-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --optimizer "AdamW" \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --hp_feats_path $hp_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \
        --window_size $window_size \

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

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl"
run_name="2-2_d_h_womp-logits-nch2.4_ws30-ss5-adamw"
hid_channels="64 512"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.2-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --optimizer "AdamW" \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --hp_feats_path $hp_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \
        --window_size $window_size \

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

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl"
run_name="2-2_d_h_womp-logits-nch2.2_ws30-ss5-lr0.0005-adamw"
hid_channels="8 512"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.2-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --optimizer "AdamW" \
    --lr 0.0005 \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --hp_feats_path $hp_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \
        --window_size $window_size \

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

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl"
run_name="2-2_d_h_womp-logits-nch2.3_ws30-ss5-lr0.0005-adamw"
hid_channels="32 512"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.2-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --optimizer "AdamW" \
    --lr 0.0005 \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --hp_feats_path $hp_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \
        --window_size $window_size \

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

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl"
run_name="2-2_d_h_womp-logits-nch2.4_ws30-ss5-lr0.0005-adamw"
hid_channels="64 512"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    python ../src/2.2-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --optimizer "AdamW" \
    --lr 0.0005 \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --hp_feats_path $hp_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \
        --window_size $window_size \

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
--message "run-com_h3 finish!" \
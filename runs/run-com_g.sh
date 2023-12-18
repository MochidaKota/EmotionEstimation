RUN_NUM=8
target_emo="comfort"
gpu_id="0"
labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize5.csv"
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize5.csv"
####################################################################################################
use_feat_list="0 1 0"
batchnorm="True"
val_phase="True"
maxpool="True"
window_size=30
n_epochs=10
lr=0.005
gaze_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/gaze(deg).pkl"

run_name="2-2_c_g_nch2_ws30-ss5-lr0.005"
hid_channels="8 32"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --batchnorm $batchnorm \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --n_epochs $n_epochs \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \

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

run_name="2-2_c_g_nch3_ws30-ss5-lr0.005"
hid_channels="8 32 64"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --batchnorm $batchnorm \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --n_epochs $n_epochs \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \

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

gaze_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/gaze(deg)_delta1.pkl"

run_name="2-2_c_g_delta1-nch2_ws30-ss5-lr0.005"
hid_channels="8 32"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --batchnorm $batchnorm \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --n_epochs $n_epochs \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \

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

run_name="2-2_c_g_delta1-nch3_ws30-ss5-lr0.005"
hid_channels="8 32 64"

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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --batchnorm $batchnorm \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --n_epochs $n_epochs \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \

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

gaze_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_pitchyaw_logits.pkl"

run_name="2-2_c_g_logits-nch2_ws30-ss5-lr0.005"
hid_channels="256 512"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --batchnorm $batchnorm \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --n_epochs $n_epochs \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \

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

run_name="2-2_c_g_logits-nch3_ws30-ss5-lr0.005"
hid_channels="256 512 1024"

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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --batchnorm $batchnorm \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --n_epochs $n_epochs \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \

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

gaze_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_feature.pkl"

run_name="2-2_c_g_feats-nch2_ws30-ss5-lr0.005"
hid_channels="256 512"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --batchnorm $batchnorm \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --n_epochs $n_epochs \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \

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

run_name="2-2_c_g_feats-nch3_ws30-ss5-lr0.005"
hid_channels="256 512 1024"

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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --batchnorm $batchnorm \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --n_epochs $n_epochs \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --batchnorm $batchnorm \
        --maxpool $maxpool \
        --hid_channels $hid_channels \

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

use_feat_list="0 1 0"
classifier="LSTM"
hidden_dim=256
val_phase="True"
window_size=30
n_epochs=10
lr=0.005
gaze_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_feature.pkl"

run_name="2-2-1_c_g_feats_ws30-ss5-lr0.005"
bidirectional="False"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --classifier $classifier \
    --bidirectional $bidirectional \
    --hidden_dim $hidden_dim \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --classifier $classifier \
        --bidirectional $bidirectional \
        --hidden_dim $hidden_dim \

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

run_name="2-2-2_c_g_feats_ws30-ss5-lr0.005"
bidirectional="True"

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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --classifier $classifier \
    --bidirectional $bidirectional \
    --hidden_dim $hidden_dim \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --classifier $classifier \
        --bidirectional $bidirectional \
        --hidden_dim $hidden_dim \

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

gaze_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_pitchyaw_logits.pkl"

run_name="2-2-1_c_g_logits_ws30-ss5-lr0.005"
bidirectional="False"

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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --classifier $classifier \
    --bidirectional $bidirectional \
    --hidden_dim $hidden_dim \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --classifier $classifier \
        --bidirectional $bidirectional \
        --hidden_dim $hidden_dim \

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

run_name="2-2-2_c_g_logits_ws30-ss5-lr0.005"
bidirectional="True"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --classifier $classifier \
    --bidirectional $bidirectional \
    --hidden_dim $hidden_dim \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --classifier $classifier \
        --bidirectional $bidirectional \
        --hidden_dim $hidden_dim \

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

gaze_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/gaze(deg).pkl"

run_name="2-2-1_c_g_ws30-ss5-lr0.005"
bidirectional="False"

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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --classifier $classifier \
    --bidirectional $bidirectional \
    --hidden_dim $hidden_dim \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --classifier $classifier \
        --bidirectional $bidirectional \
        --hidden_dim $hidden_dim \

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

run_name="2-2-2_c_g_ws30-ss5-lr0.005"
bidirectional="True"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --classifier $classifier \
    --bidirectional $bidirectional \
    --hidden_dim $hidden_dim \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --classifier $classifier \
        --bidirectional $bidirectional \
        --hidden_dim $hidden_dim \

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

gaze_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/gaze(deg)_delta1.pkl"

run_name="2-2-1_c_g_delta1_ws30-ss5-lr0.005"
bidirectional="False"

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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --classifier $classifier \
    --bidirectional $bidirectional \
    --hidden_dim $hidden_dim \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --classifier $classifier \
        --bidirectional $bidirectional \
        --hidden_dim $hidden_dim \

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

run_name="2-2-2_c_g_delta1_ws30-ss5-lr0.005"
bidirectional="True"


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
    --gaze_feats_path $gaze_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --val_phase $val_phase \
    --window_size $window_size \
    --n_epochs $n_epochs \
    --classifier $classifier \
    --bidirectional $bidirectional \
    --hidden_dim $hidden_dim \
    --lr $lr \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/2.2-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gaze_feats_path $gaze_feats_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --classifier $classifier \
        --bidirectional $bidirectional \
        --hidden_dim $hidden_dim \

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
--message "run-com_g finish!" \
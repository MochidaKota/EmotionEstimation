RUN_NUM=6
target_emo="discomfort"
gpu_id="0"
labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_nomixed_seq_labels_wsize30-ssize5.csv"
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_nomixed_seq_video_name_list_wsize30-ssize5.csv"
####################################################################################################
use_feat_list="0 0 1"
batchnorm="True"
maxpool="True"
val_phase="True"
window_size=30
n_epochs=10
lr=0.005

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/headpose_py.pkl"
run_name="2-2_d_h_nch2_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
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

run_name="2-2_d_h_nch3_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
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

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/headpose_py_delta1.pkl"
run_name="2-2_d_h_delta1-nch2_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
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

run_name="2-2_d_h_delta1-nch3_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
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
run_name="2-2_d_h_logits-nch2_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
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

run_name="2-2_d_h_logits-nch3_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
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

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_feature.pkl"
run_name="2-2_d_h_feats-nch2_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
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

run_name="2-2_d_h_feats-nch3_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --batchnorm $batchnorm \
    --val_phase $val_phase \
    --maxpool $maxpool \
    --hid_channels $hid_channels \
    --window_size $window_size \
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

use_feat_list="0 0 1"
classifier="LSTM"
hidden_dim=256
val_phase="True"
window_size=30
n_epochs=10
lr=0.005

hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_feature.pkl"

run_name="2-2-1_d_h_feats_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
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
        --hp_feats_path $hp_feats_path \
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

run_name="2-2-2_d_h_feats_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
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
        --hp_feats_path $hp_feats_path \
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

run_name="2-2-1_d_h_logits_ws30-ss5-lr0.005"
bidirectional="False"
hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/6DRepNet_logits.pkl"


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
        --hp_feats_path $hp_feats_path \
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

run_name="2-2-2_d_h_logits_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
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
        --hp_feats_path $hp_feats_path \
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

run_name="2-2-1_d_h_ws30-ss5-lr0.005"
bidirectional="False"
hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/headpose_py.pkl"


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
        --hp_feats_path $hp_feats_path \
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

run_name="2-2-2_d_h_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
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
        --hp_feats_path $hp_feats_path \
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

run_name="2-2-1_d_h_delta1_ws30-ss5-lr0.005"
bidirectional="False"
hp_feats_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/headpose_py_delta1.pkl"


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
        --hp_feats_path $hp_feats_path \
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

run_name="2-2-2_d_h_delta1_ws30-ss5-lr0.005"
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
    --hp_feats_path $hp_feats_path \
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
        --hp_feats_path $hp_feats_path \
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
--message "run-dis_h finish!" \
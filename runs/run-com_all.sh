RUN_NUM=8
target_emo="comfort"
gpu_id="0"
labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_labels_wsize30-ssize5.csv"
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/comfort_nomixed_seq_video_name_list_wsize30-ssize5.csv"
####################################################################################################

hidden_dims="2048 512"
attpool_hidden_dims="512"
hid_channels="256 512"
n_epochs=10
save_feat="True"

run_name="4_c_a"
use_feat_list="1 0 0"
classifier="MLP"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    # python ../src/4.0-train-model.py \
    # --run_name $run_name \
    # --fold $i \
    # --target_emo $target_emo \
    # --labels_path $labels_path \
    # --video_name_list_path $video_name_list_path \
    # --gpu_id $gpu_id \
    # --use_feat_list $use_feat_list \
    # --hidden_dims $hidden_dims \
    # --attpool_hidden_dims $attpool_hidden_dims \
    # --hid_channels $hid_channels \
    # --classifier $classifier \
    # --n_epochs $n_epochs \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/4.0-test-model.py \
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
        --hid_channels $hid_channels \
        --classifier $classifier \
        --save_feat $save_feat \

    done
    
done


# for ((i=1; i<=$n_epochs; i++))
# do
#     python ../src/utils/culc-cv_result.py \
#     --run_name $run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

# done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="4_c_g"
use_feat_list="0 1 0"
hid_channels="256 512"
classifier="1DCNN"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    # python ../src/4.0-train-model.py \
    # --run_name $run_name \
    # --fold $i \
    # --target_emo $target_emo \
    # --labels_path $labels_path \
    # --video_name_list_path $video_name_list_path \
    # --gpu_id $gpu_id \
    # --use_feat_list $use_feat_list \
    # --hidden_dims $hidden_dims \
    # --attpool_hidden_dims $attpool_hidden_dims \
    # --hid_channels $hid_channels \
    # --classifier $classifier \
    # --n_epochs $n_epochs \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/4.0-test-model.py \
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
        --hid_channels $hid_channels \
        --classifier $classifier \
        --save_feat $save_feat \

    done
    
done


# for ((i=1; i<=$n_epochs; i++))
# do
#     python ../src/utils/culc-cv_result.py \
#     --run_name $run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

# done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

run_name="4_c_h"
use_feat_list="0 0 1"
hid_channels="8 512"
classifier="1DCNN"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do
    # python ../src/4.0-train-model.py \
    # --run_name $run_name \
    # --fold $i \
    # --target_emo $target_emo \
    # --labels_path $labels_path \
    # --video_name_list_path $video_name_list_path \
    # --gpu_id $gpu_id \
    # --use_feat_list $use_feat_list \
    # --hidden_dims $hidden_dims \
    # --attpool_hidden_dims $attpool_hidden_dims \
    # --hid_channels $hid_channels \
    # --classifier $classifier \
    # --n_epochs $n_epochs \

    for ((j=1; j<=$n_epochs; j++))
    do
        python ../src/4.0-test-model.py \
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
        --hid_channels $hid_channels \
        --classifier $classifier \
        --save_feat $save_feat \

    done
    
done


# for ((i=1; i<=$n_epochs; i++))
# do
#     python ../src/utils/culc-cv_result.py \
#     --run_name $run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

# done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

python ../src/utils/notify-slack.py \
--message "com_all finish!" \
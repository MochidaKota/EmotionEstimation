RUN_NUM=5
target_emo="discomfort"
gpu_id="0"
window_size=30
n_epochs=10
save_feat="True"
labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_labels_wsize$window_size-ssize3.csv"
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_video_name_list_wsize$window_size-ssize3.csv"
all_labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/seq_labels(video1-25)_ver2_wsize$window_size-ssize3-th5e-01.csv"
all_video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_video_name_list_wsize$window_size-ssize3.csv"
####################################################################################################

# run_name="4_d_a_1dcnn_ws30-ss3"
# other_run_name="$run_name(all)"
# use_feat_list="1 0 0"
# hidden_dims="2048 512"
# hid_channels="2048 512"
# classifier="1DCNN"

# echo ------------------- $run_name start! -------------------

# python ../src/utils/notify-slack.py \
# --message "$run_name start!" \

# for ((i=1; i<=RUN_NUM; i++))
# do

#     python ../src/4.0-train-model.py \
#     --run_name $run_name \
#     --fold $i \
#     --target_emo $target_emo \
#     --labels_path $labels_path \
#     --video_name_list_path $video_name_list_path \
#     --gpu_id $gpu_id \
#     --use_feat_list $use_feat_list \
#     --window_size $window_size \
#     --hidden_dims $hidden_dims \
#     --hid_channels $hid_channels \
#     --classifier $classifier \
#     --n_epochs $n_epochs \

#     for ((j=1; j<=$n_epochs; j++))
#     do

#         python ../src/4.0-test-model.py \
#         --run_name $run_name \
#         --fold $i \
#         --target_emo $target_emo \
#         --target_epoch $j \
#         --labels_path $labels_path \
#         --video_name_list_path $video_name_list_path \
#         --gpu_id $gpu_id \
#         --use_feat_list $use_feat_list \
#         --window_size $window_size \
#         --hidden_dims $hidden_dims \
#         --hid_channels $hid_channels \
#         --classifier $classifier \
#         --save_feat $save_feat \

#         python ../src/4.0-test-model.py \
#         --run_name $run_name \
#         --fold $i \
#         --target_emo $target_emo \
#         --target_epoch $j \
#         --labels_path $all_labels_path \
#         --video_name_list_path $all_video_name_list_path \
#         --gpu_id $gpu_id \
#         --use_feat_list $use_feat_list \
#         --window_size $window_size \
#         --hidden_dims $hidden_dims \
#         --hid_channels $hid_channels \
#         --classifier $classifier \
#         --save_feat $save_feat \
#         --other_run_name $other_run_name \

#     done
    
# done


# for ((i=1; i<=$n_epochs; i++))
# do
#     python ../src/utils/calc-cv_result.py \
#     --run_name $run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

#     python ../src/utils/calc-cv_result.py \
#     --run_name $other_run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

# done

# python ../src/utils/notify-slack.py \
# --message "$run_name finish!" \

####################################################################################################

# run_name="4_d_g_ws30-ss3"
# other_run_name="$run_name(all)"
# use_feat_list="0 1 0"
# hid_channels="256 512"
# classifier="1DCNN"

# echo ------------------- $run_name start! -------------------

# python ../src/utils/notify-slack.py \
# --message "$run_name start!" \

# for ((i=1; i<=RUN_NUM; i++))
# do

#     python ../src/4.0-train-model.py \
#     --run_name $run_name \
#     --fold $i \
#     --target_emo $target_emo \
#     --labels_path $labels_path \
#     --video_name_list_path $video_name_list_path \
#     --gpu_id $gpu_id \
#     --use_feat_list $use_feat_list \
#     --window_size $window_size \
#     --hidden_dims $hidden_dims \
#     --hid_channels $hid_channels \
#     --classifier $classifier \
#     --n_epochs $n_epochs \

#     for ((j=1; j<=$n_epochs; j++))
#     do

#         python ../src/4.0-test-model.py \
#         --run_name $run_name \
#         --fold $i \
#         --target_emo $target_emo \
#         --target_epoch $j \
#         --labels_path $labels_path \
#         --video_name_list_path $video_name_list_path \
#         --gpu_id $gpu_id \
#         --use_feat_list $use_feat_list \
#         --window_size $window_size \
#         --hidden_dims $hidden_dims \
#         --hid_channels $hid_channels \
#         --classifier $classifier \
#         --save_feat $save_feat \

#         python ../src/4.0-test-model.py \
#         --run_name $run_name \
#         --fold $i \
#         --target_emo $target_emo \
#         --target_epoch $j \
#         --labels_path $all_labels_path \
#         --video_name_list_path $all_video_name_list_path \
#         --gpu_id $gpu_id \
#         --use_feat_list $use_feat_list \
#         --window_size $window_size \
#         --hidden_dims $hidden_dims \
#         --hid_channels $hid_channels \
#         --classifier $classifier \
#         --save_feat $save_feat \
#         --other_run_name $other_run_name \

#     done
    
# done


# for ((i=1; i<=$n_epochs; i++))
# do
#     python ../src/utils/calc-cv_result.py \
#     --run_name $run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

#     python ../src/utils/calc-cv_result.py \
#     --run_name $other_run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

# done

# python ../src/utils/notify-slack.py \
# --message "$run_name finish!" \

# ####################################################################################################

run_name="4_d_h_ws30-ss3"
other_run_name="$run_name(all)"
use_feat_list="0 0 1"
hid_channels="8 512"
classifier="1DCNN"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do

    python ../src/4.0-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --use_feat_list $use_feat_list \
    --window_size $window_size \
    --hidden_dims $hidden_dims \
    --hid_channels $hid_channels \
    --classifier $classifier \
    --n_epochs $n_epochs \

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
        --window_size $window_size \
        --hidden_dims $hidden_dims \
        --hid_channels $hid_channels \
        --classifier $classifier \
        --save_feat $save_feat \

        python ../src/4.0-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $all_labels_path \
        --video_name_list_path $all_video_name_list_path \
        --gpu_id $gpu_id \
        --use_feat_list $use_feat_list \
        --window_size $window_size \
        --hidden_dims $hidden_dims \
        --hid_channels $hid_channels \
        --classifier $classifier \
        --save_feat $save_feat \
        --other_run_name $other_run_name \

    done
    
done


for ((i=1; i<=$n_epochs; i++))
do
    python ../src/utils/calc-cv_result.py \
    --run_name $run_name \
    --target_emo $target_emo \
    --target_epoch $i \

    python ../src/utils/calc-cv_result.py \
    --run_name $other_run_name \
    --target_emo $target_emo \
    --target_epoch $i \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

####################################################################################################

# python ../src/utils/notify-slack.py \
# --message "dis_all finish!" \
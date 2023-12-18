RUN_NUM=5
target_emo="discomfort"
gpu_id="0"
window_size=300
n_epochs=10
labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_labels_wsize$window_size-ssize3.csv"
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_video_name_list_wsize$window_size-ssize3.csv"
all_labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/seq_labels(video1-25)_ver2_wsize$window_size-ssize3-th5e-01.csv"
all_video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_video_name_list_wsize$window_size-ssize3.csv"

au_run_name="4_d_a_austream2_ws300-ss3"
au_epoch=5
au_stream="AUStream"
gaze_run_name="4_d_g_ws300-ss3"
gaze_epoch=6
hp_run_name="4_d_h_ws300-ss3"
hp_epoch=5
####################################################################################################

# run_name="4_d_ag_austream2_ws300-ss3"
# other_run_name="$run_name(all)"
# use_feat_list="1 1 0"
# integrate_dim="512"
# integrated_hidden_dims="512 128"
# stream_mixer_hidden_dims=""
# summation="False"
# is_stream_mixer="False"
# integrate_point="mid"
# stream_mixer_input="mid"

# echo ------------------- $run_name start! -------------------

# python ../src/utils/notify-slack.py \
# --message "$run_name start!" \

# for ((i=1; i<=RUN_NUM; i++))
# do

#     python ../src/4.1-train-model.py \
#     --run_name $run_name \
#     --fold $i \
#     --target_emo $target_emo \
#     --labels_path $labels_path \
#     --video_name_list_path $video_name_list_path \
#     --gpu_id $gpu_id \
#     --window_size $window_size \
#     --use_feat_list $use_feat_list \
#     --integrate_dim $integrate_dim \
#     --integrated_hidden_dims $integrated_hidden_dims \
#     --integrate_point $integrate_point \
#     --summation $summation \
#     --is_stream_mixer $is_stream_mixer \
#     --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#     --stream_mixer_input $stream_mixer_input \
#     --au_run_name $au_run_name \
#     --au_epoch $au_epoch \
#     --au_stream $au_stream \
#     --gaze_run_name $gaze_run_name \
#     --gaze_epoch $gaze_epoch \
#     --hp_run_name $hp_run_name \
#     --hp_epoch $hp_epoch \
#     --n_epochs $n_epochs \

#     for ((j=1; j<=$n_epochs; j++))
#     do

#         python ../src/4.1-test-model.py \
#         --run_name $run_name \
#         --fold $i \
#         --target_emo $target_emo \
#         --target_epoch $j \
#         --labels_path $labels_path \
#         --video_name_list_path $video_name_list_path \
#         --gpu_id $gpu_id \
#         --window_size $window_size \
#         --use_feat_list $use_feat_list \
#         --integrate_dim $integrate_dim \
#         --integrated_hidden_dims $integrated_hidden_dims \
#         --integrate_point $integrate_point \
#         --summation $summation \
#         --is_stream_mixer $is_stream_mixer \
#         --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#         --stream_mixer_input $stream_mixer_input \
#         --au_run_name $au_run_name \
#         --au_epoch $au_epoch \
#         --au_stream $au_stream \
#         --gaze_run_name $gaze_run_name \
#         --gaze_epoch $gaze_epoch \
#         --hp_run_name $hp_run_name \
#         --hp_epoch $hp_epoch \

#         python ../src/4.1-test-model.py \
#         --run_name $run_name \
#         --fold $i \
#         --target_emo $target_emo \
#         --target_epoch $j \
#         --labels_path $all_labels_path \
#         --video_name_list_path $all_video_name_list_path \
#         --gpu_id $gpu_id \
#         --window_size $window_size \
#         --use_feat_list $use_feat_list \
#         --integrate_dim $integrate_dim \
#         --integrated_hidden_dims $integrated_hidden_dims \
#         --integrate_point $integrate_point \
#         --summation $summation \
#         --is_stream_mixer $is_stream_mixer \
#         --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#         --stream_mixer_input $stream_mixer_input \
#         --au_run_name $au_run_name \
#         --au_epoch $au_epoch \
#         --au_stream $au_stream \
#         --gaze_run_name $gaze_run_name \
#         --gaze_epoch $gaze_epoch \
#         --hp_run_name $hp_run_name \
#         --hp_epoch $hp_epoch \
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

# run_name="4_d_ag_wsum-austream2_ws300-ss3"
# other_run_name="$run_name(all)"
# use_feat_list="1 1 0"
# integrate_dim="512"
# integrated_hidden_dims="128"
# stream_mixer_hidden_dims="512 128"
# summation="True"
# is_stream_mixer="True"
# integrate_point="mid"
# stream_mixer_input="mid"

# echo ------------------- $run_name start! -------------------

# python ../src/utils/notify-slack.py \
# --message "$run_name start!" \

# for ((i=1; i<=RUN_NUM; i++))
# do

#     python ../src/4.1-train-model.py \
#     --run_name $run_name \
#     --fold $i \
#     --target_emo $target_emo \
#     --labels_path $labels_path \
#     --video_name_list_path $video_name_list_path \
#     --gpu_id $gpu_id \
#     --window_size $window_size \
#     --use_feat_list $use_feat_list \
#     --integrate_dim $integrate_dim \
#     --integrated_hidden_dims $integrated_hidden_dims \
#     --integrate_point $integrate_point \
#     --summation $summation \
#     --is_stream_mixer $is_stream_mixer \
#     --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#     --stream_mixer_input $stream_mixer_input \
#     --au_run_name $au_run_name \
#     --au_epoch $au_epoch \
#     --au_stream $au_stream \
#     --gaze_run_name $gaze_run_name \
#     --gaze_epoch $gaze_epoch \
#     --hp_run_name $hp_run_name \
#     --hp_epoch $hp_epoch \
#     --n_epochs $n_epochs \

#     for ((j=1; j<=$n_epochs; j++))
#     do

#         python ../src/4.1-test-model.py \
#         --run_name $run_name \
#         --fold $i \
#         --target_emo $target_emo \
#         --target_epoch $j \
#         --labels_path $labels_path \
#         --video_name_list_path $video_name_list_path \
#         --gpu_id $gpu_id \
#         --window_size $window_size \
#         --use_feat_list $use_feat_list \
#         --integrate_dim $integrate_dim \
#         --integrated_hidden_dims $integrated_hidden_dims \
#         --integrate_point $integrate_point \
#         --summation $summation \
#         --is_stream_mixer $is_stream_mixer \
#         --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#         --stream_mixer_input $stream_mixer_input \
#         --au_run_name $au_run_name \
#         --au_epoch $au_epoch \
#         --au_stream $au_stream \
#         --gaze_run_name $gaze_run_name \
#         --gaze_epoch $gaze_epoch \
#         --hp_run_name $hp_run_name \
#         --hp_epoch $hp_epoch \

#         python ../src/4.1-test-model.py \
#         --run_name $run_name \
#         --fold $i \
#         --target_emo $target_emo \
#         --target_epoch $j \
#         --labels_path $all_labels_path \
#         --video_name_list_path $all_video_name_list_path \
#         --gpu_id $gpu_id \
#         --window_size $window_size \
#         --use_feat_list $use_feat_list \
#         --integrate_dim $integrate_dim \
#         --integrated_hidden_dims $integrated_hidden_dims \
#         --integrate_point $integrate_point \
#         --summation $summation \
#         --is_stream_mixer $is_stream_mixer \
#         --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#         --stream_mixer_input $stream_mixer_input \
#         --au_run_name $au_run_name \
#         --au_epoch $au_epoch \
#         --au_stream $au_stream \
#         --gaze_run_name $gaze_run_name \
#         --gaze_epoch $gaze_epoch \
#         --hp_run_name $hp_run_name \
#         --hp_epoch $hp_epoch \
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

run_name="4_d_ag_wsum-intlogits-austream2_ws300-ss3"
other_run_name="$run_name(all)"
use_feat_list="1 1 0"
integrate_dim="512"
integrated_hidden_dims="128"
integrate_point="logits"
summation="True"
is_stream_mixer="True"
stream_mixer_hidden_dims="512 128"
stream_mixer_input="mid"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do

    python ../src/4.1-train-model.py \
    --run_name $run_name \
    --fold $i \
    --target_emo $target_emo \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --gpu_id $gpu_id \
    --window_size $window_size \
    --use_feat_list $use_feat_list \
    --integrate_dim $integrate_dim \
    --integrated_hidden_dims $integrated_hidden_dims \
    --integrate_point $integrate_point \
    --summation $summation \
    --is_stream_mixer $is_stream_mixer \
    --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
    --stream_mixer_input $stream_mixer_input \
    --au_run_name $au_run_name \
    --au_epoch $au_epoch \
    --au_stream $au_stream \
    --gaze_run_name $gaze_run_name \
    --gaze_epoch $gaze_epoch \
    --hp_run_name $hp_run_name \
    --hp_epoch $hp_epoch \
    --n_epochs $n_epochs \

    for ((j=1; j<=$n_epochs; j++))
    do

        python ../src/4.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --gpu_id $gpu_id \
        --window_size $window_size \
        --use_feat_list $use_feat_list \
        --integrate_dim $integrate_dim \
        --integrated_hidden_dims $integrated_hidden_dims \
        --integrate_point $integrate_point \
        --summation $summation \
        --is_stream_mixer $is_stream_mixer \
        --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
        --stream_mixer_input $stream_mixer_input \
        --au_run_name $au_run_name \
        --au_epoch $au_epoch \
        --au_stream $au_stream \
        --gaze_run_name $gaze_run_name \
        --gaze_epoch $gaze_epoch \
        --hp_run_name $hp_run_name \
        --hp_epoch $hp_epoch \

        python ../src/4.1-test-model.py \
        --run_name $run_name \
        --fold $i \
        --target_emo $target_emo \
        --target_epoch $j \
        --labels_path $all_labels_path \
        --video_name_list_path $all_video_name_list_path \
        --gpu_id $gpu_id \
        --window_size $window_size \
        --use_feat_list $use_feat_list \
        --integrate_dim $integrate_dim \
        --integrated_hidden_dims $integrated_hidden_dims \
        --integrate_point $integrate_point \
        --summation $summation \
        --is_stream_mixer $is_stream_mixer \
        --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
        --stream_mixer_input $stream_mixer_input \
        --au_run_name $au_run_name \
        --au_epoch $au_epoch \
        --au_stream $au_stream \
        --gaze_run_name $gaze_run_name \
        --gaze_epoch $gaze_epoch \
        --hp_run_name $hp_run_name \
        --hp_epoch $hp_epoch \
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

python ../src/utils/notify-slack.py \
--message "dis_all2 finished!" \
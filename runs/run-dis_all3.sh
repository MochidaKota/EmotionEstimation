RUN_NUM=6
target_emo="discomfort"
gpu_id="0"
labels_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_nomixed_seq_labels_wsize30-ssize5.csv"
video_name_list_path="/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/discomfort_nomixed_seq_video_name_list_wsize30-ssize5.csv"
####################################################################################################

au_run_name="4_d_a"
au_epoch=10
gaze_run_name="4_d_g"
gaze_epoch=10
hp_run_name="4_d_h"
hp_epoch=5
n_epochs=10

integrated_hidden_dims="128"
stream_mixer_hidden_dims="512 128"
summation="True"
is_stream_mixer="True"
ew_product="False"
arith_mean="False"
is_standardization="False"
is_calibrator="False"
freeze_stream="True"
integrate_point="logits"

run_name="4_d_agh_wsum-intlogits"
use_feat_list="1 1 1"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

# for ((i=1; i<=RUN_NUM; i++))
# do
#     python ../src/4.1-train-model.py \
#     --run_name $run_name \
#     --fold $i \
#     --target_emo $target_emo \
#     --labels_path $labels_path \
#     --video_name_list_path $video_name_list_path \
#     --gpu_id $gpu_id \
#     --use_feat_list $use_feat_list \
#     --integrated_hidden_dims $integrated_hidden_dims \
#     --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#     --au_run_name $au_run_name \
#     --au_epoch $au_epoch \
#     --gaze_run_name $gaze_run_name \
#     --gaze_epoch $gaze_epoch \
#     --hp_run_name $hp_run_name \
#     --hp_epoch $hp_epoch \
#     --n_epochs $n_epochs \
#     --summation $summation \
#     --is_stream_mixer $is_stream_mixer \
#     --ew_product $ew_product \
#     --arith_mean $arith_mean \
#     --is_standardization $is_standardization \
#     --is_calibrator $is_calibrator \
#     --freeze_stream $freeze_stream \
#     --integrate_point $integrate_point \

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
#         --use_feat_list $use_feat_list \
#         --integrated_hidden_dims $integrated_hidden_dims \
#         --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#         --au_run_name $au_run_name \
#         --au_epoch $au_epoch \
#         --gaze_run_name $gaze_run_name \
#         --gaze_epoch $gaze_epoch \
#         --hp_run_name $hp_run_name \
#         --hp_epoch $hp_epoch \
#         --summation $summation \
#         --is_stream_mixer $is_stream_mixer \
#         --ew_product $ew_product \
#         --arith_mean $arith_mean \
#         --is_standardization $is_standardization \
#         --is_calibrator $is_calibrator \
#         --freeze_stream $freeze_stream \
#         --integrate_point $integrate_point \

#     done
    
# done


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

# integrated_hidden_dims="128"
# stream_mixer_hidden_dims="512 128"
# summation="True"
# is_stream_mixer="True"
# ew_product="False"
# arith_mean="False"
# is_standardization="False"
# is_calibrator="True"

# run_name="4_d_agh_wsum-cal"
# use_feat_list="1 1 1"

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
#     --use_feat_list $use_feat_list \
#     --integrated_hidden_dims $integrated_hidden_dims \
#     --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#     --au_run_name $au_run_name \
#     --au_epoch $au_epoch \
#     --gaze_run_name $gaze_run_name \
#     --gaze_epoch $gaze_epoch \
#     --hp_run_name $hp_run_name \
#     --hp_epoch $hp_epoch \
#     --n_epochs $n_epochs \
#     --summation $summation \
#     --is_stream_mixer $is_stream_mixer \
#     --ew_product $ew_product \
#     --arith_mean $arith_mean \
#     --is_standardization $is_standardization \
#     --is_calibrator $is_calibrator \

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
#         --use_feat_list $use_feat_list \
#         --integrated_hidden_dims $integrated_hidden_dims \
#         --stream_mixer_hidden_dims $stream_mixer_hidden_dims \
#         --au_run_name $au_run_name \
#         --au_epoch $au_epoch \
#         --gaze_run_name $gaze_run_name \
#         --gaze_epoch $gaze_epoch \
#         --hp_run_name $hp_run_name \
#         --hp_epoch $hp_epoch \
#         --summation $summation \
#         --is_stream_mixer $is_stream_mixer \
#         --ew_product $ew_product \
#         --arith_mean $arith_mean \
#         --is_standardization $is_standardization \
#         --is_calibrator $is_calibrator \

#     done
    
# done


# for ((i=1; i<=$n_epochs; i++))
# do
#     python ../src/utils/culc-cv_result.py \
#     --run_name $run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

# done

# python ../src/utils/notify-slack.py \
# --message "$run_name finish!" \

####################################################################################################

# run_name="4_d_gh_std_actLR"
# use_feat_list="0 1 1"

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
#     --use_feat_list $use_feat_list \
#     --integrated_hidden_dims $integrated_hidden_dims \
#     --au_run_name $au_run_name \
#     --au_epoch $au_epoch \
#     --gaze_run_name $gaze_run_name \
#     --gaze_epoch $gaze_epoch \
#     --hp_run_name $hp_run_name \
#     --hp_epoch $hp_epoch \
#     --n_epochs $n_epochs \
#     --summation $summation \
#     --is_stream_mixer $is_stream_mixer \
#     --ew_product $ew_product \
#     --arith_mean $arith_mean \
#     --is_standardization $is_standardization \
#     --activation "LeakyReLU" \

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
#         --use_feat_list $use_feat_list \
#         --integrated_hidden_dims $integrated_hidden_dims \
#         --au_run_name $au_run_name \
#         --au_epoch $au_epoch \
#         --gaze_run_name $gaze_run_name \
#         --gaze_epoch $gaze_epoch \
#         --hp_run_name $hp_run_name \
#         --hp_epoch $hp_epoch \
#         --summation $summation \
#         --is_stream_mixer $is_stream_mixer \
#         --ew_product $ew_product \
#         --arith_mean $arith_mean \
#         --is_standardization $is_standardization \
#         --activation "LeakyReLU" \

#     done
    
# done


# for ((i=1; i<=$n_epochs; i++))
# do
#     python ../src/utils/culc-cv_result.py \
#     --run_name $run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

# done

# python ../src/utils/notify-slack.py \
# --message "$run_name finish!" \

# ####################################################################################################

# run_name="4_d_agh_std_actLR"
# use_feat_list="1 1 1"

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
#     --use_feat_list $use_feat_list \
#     --integrated_hidden_dims $integrated_hidden_dims \
#     --au_run_name $au_run_name \
#     --au_epoch $au_epoch \
#     --gaze_run_name $gaze_run_name \
#     --gaze_epoch $gaze_epoch \
#     --hp_run_name $hp_run_name \
#     --hp_epoch $hp_epoch \
#     --n_epochs $n_epochs \
#     --summation $summation \
#     --is_stream_mixer $is_stream_mixer \
#     --ew_product $ew_product \
#     --arith_mean $arith_mean \
#     --is_standardization $is_standardization \
#     --activation "LeakyReLU" \

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
#         --use_feat_list $use_feat_list \
#         --integrated_hidden_dims $integrated_hidden_dims \
#         --au_run_name $au_run_name \
#         --au_epoch $au_epoch \
#         --gaze_run_name $gaze_run_name \
#         --gaze_epoch $gaze_epoch \
#         --hp_run_name $hp_run_name \
#         --hp_epoch $hp_epoch \
#         --summation $summation \
#         --is_stream_mixer $is_stream_mixer \
#         --ew_product $ew_product \
#         --arith_mean $arith_mean \
#         --is_standardization $is_standardization \
#         --activation "LeakyReLU" \

#     done
    
# done


# for ((i=1; i<=$n_epochs; i++))
# do
#     python ../src/utils/culc-cv_result.py \
#     --run_name $run_name \
#     --target_emo $target_emo \
#     --target_epoch $i \

# done

# python ../src/utils/notify-slack.py \
# --message "$run_name finish!" \

# ####################################################################################################

# python ../src/utils/notify-slack.py \
# --message "dis_all3 finished!" \
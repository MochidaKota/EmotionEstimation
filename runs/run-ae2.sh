RUN_NUM=8
gpu_id="0"
num_epochs=20
########################################################################################################################

run_name="5_d_g"
feats_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_feature.pkl'
val_phase="True"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do

    labels_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/label_video18-25_Gaze_onlypositive.csv'
    video_name_list_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/video_name_list_video18-25_Gaze_onlypositive.csv'

    # echo ------ fold $i start! --------  

    # python ../src/5.0-train-model.py \
    # --run_name $run_name \
    # --gpu_id $gpu_id \
    # --fold $i \
    # --num_epochs $num_epochs \
    # --val_phase $val_phase \
    # --labels_path $labels_path \
    # --video_name_list_path $video_name_list_path \
    # --feats_path $feats_path \
    # --lr 0.05 \

    labels_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au-gaze-hp(video1-25)_temp_Gaze.csv'
    video_name_list_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au(video1-25)-video_name_list_temp.csv'

    for ((j=1; j<=$num_epochs; j++))
    do

        echo ------ fold $i epoch $j --------

        python ../src/5.0-test-model.py \
        --run_name $run_name \
        --gpu_id $gpu_id \
        --fold $i \
        --target_epoch $j \
        --labels_path $labels_path \
        --video_name_list_path $video_name_list_path \
        --feats_path $feats_path \

    done

done 

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

echo ------------------- $run_name finish! -------------------

########################################################################################################################

run_name="5_d_g_all"
labels_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/label_video18-25_Gaze_onlypositive.csv'
video_name_list_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/video_name_list_video18-25_Gaze_onlypositive.csv'
feats_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_feature.pkl'
val_phase="False"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

# python ../src/5.0-train-model.py \
# --run_name $run_name \
# --gpu_id $gpu_id \
# --fold 0 \
# --num_epochs $num_epochs \
# --val_phase $val_phase \
# --labels_path $labels_path \
# --video_name_list_path $video_name_list_path \
# --feats_path $feats_path \
# --lr 0.05 \

labels_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au-gaze-hp(video1-25)_temp_Gaze.csv'
video_name_list_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au(video1-25)-video_name_list.csv'


for ((j=1; j<=$num_epochs; j++))
do

    echo ------ epoch $j --------

    python ../src/5.0-test-model.py \
    --run_name $run_name \
    --gpu_id $gpu_id \
    --fold 0 \
    --target_epoch $j \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --feats_path $feats_path \

done

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

echo ------------------- $run_name finish! -------------------

########################################################################################################################

python ../src/utils/notify-slack.py \
--message "run-ae2 finish!" \
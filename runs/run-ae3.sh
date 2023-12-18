RUN_NUM=8
gpu_id="1"
num_epochs=20
feats_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/processed/PIMD_A/L2CSNet_pitchyaw_logits.pkl'
hidden_dim=128
output_dim=64
########################################################################################################################

run_name="5_d_g_logits"
val_phase="True"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

for ((i=1; i<=RUN_NUM; i++))
do

    labels_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/label_video18-25_Gaze_onlypositive.csv'
    video_name_list_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/video_name_list_video18-25_Gaze_onlypositive.csv'

    echo ------ fold $i start! --------  

    python ../src/5.0-train-model.py \
    --run_name $run_name \
    --gpu_id $gpu_id \
    --fold $i \
    --num_epochs $num_epochs \
    --val_phase $val_phase \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --feats_path $feats_path \
    --lr 0.1 \
    --hidden_dim $hidden_dim \
    --output_dim $output_dim \

    labels_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au-gaze-hp(video1-25)_temp_Gaze.csv'
    video_name_list_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au(video1-25)-video_name_list_temp.csv'

    python ../src/5.0-test-model.py \
    --run_name $run_name \
    --gpu_id $gpu_id \
    --fold $i \
    --target_epoch $num_epochs \
    --labels_path $labels_path \
    --video_name_list_path $video_name_list_path \
    --feats_path $feats_path \
    --hidden_dim $hidden_dim \
    --output_dim $output_dim \

done 

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

echo ------------------- $run_name finish! -------------------

########################################################################################################################

run_name="5_d_g_all-logits"
labels_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/label_video18-25_Gaze_onlypositive.csv'
video_name_list_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/video_name_list_video18-25_Gaze_onlypositive.csv'
val_phase="False"

echo ------------------- $run_name start! -------------------

python ../src/utils/notify-slack.py \
--message "$run_name start!" \

python ../src/5.0-train-model.py \
--run_name $run_name \
--gpu_id $gpu_id \
--fold 0 \
--num_epochs $num_epochs \
--val_phase $val_phase \
--labels_path $labels_path \
--video_name_list_path $video_name_list_path \
--feats_path $feats_path \
--lr 0.1 \
--hidden_dim $hidden_dim \
--output_dim $output_dim \

labels_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au-gaze-hp(video1-25)_temp_Gaze.csv'
video_name_list_path='/mnt/iot-qnap3/mochida/medical-care/emotionestimation/data/labels/PIMD_A/emo_and_au(video1-25)-video_name_list.csv'

python ../src/5.0-test-model.py \
--run_name $run_name \
--gpu_id $gpu_id \
--fold 0 \
--target_epoch $num_epochs \
--labels_path $labels_path \
--video_name_list_path $video_name_list_path \
--feats_path $feats_path \
--hidden_dim $hidden_dim \
--output_dim $output_dim \

python ../src/utils/notify-slack.py \
--message "$run_name finish!" \

echo ------------------- $run_name finish! -------------------

########################################################################################################################

python ../src/utils/notify-slack.py \
--message "run-ae3 finish!" \
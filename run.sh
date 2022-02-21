models=(
    # "x3d_m"
    # 'x3d_s'
    'x3d_xs'
    # 'slow_r50'
    "slowfast_r50"
    # "slowfast_r101"
)

gops=(
    "1"
    "2"
    "5"
    "10"
    "12"
)
crfs=(
    "0"
    "10"
    "20"
    "30"
    "40"
    "50"
)

for model in ${models[@]};do
    for gop in ${gops[@]};do
        for crf in ${crfs[@]}; do
            python dataset_imgaug.py --dataset Kinetics400 -sb val --path /mnt/HDD10TB-2/ohtani/dataset/Kinetics400_ffmpeg/gop/$gop/crf/$crf/ --model_name $model --gpu 0 --crf $crf --gop $gop;
        done
    done
done



# for model_name in ${models[@]};do
#     for path in ${data_path[@]};do
#         python dataset_imgaug_copy_2.py --path $path --model_name $model_name --q_use;
#     done
# done
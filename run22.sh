models=(
    # "x3d_m"
    # 'x3d_s'
    # 'x3d_xs'
    # 'slow_r50'
    # "x3d_l",
    "slowfast_r50"
    "slowfast_r101")
data_path=(
    "/mnt/HDD10TB-2/ohtani/dataset/Kinetics400.q_v/15/"
    "/mnt/HDD10TB-2/ohtani/dataset/Kinetics400.q_v/20/"
    # "/mnt/HDD10TB-2/ohtani/dataset/Kinetics400.q_v/25/"
    # "/mnt/HDD10TB-2/ohtani/dataset/Kinetics400.q_v/30/"
    # "/mnt/HDD10TB-2/ohtani/dataset/Kinetics400.q_v/35/"
    # "/mnt/HDD10TB-2/ohtani/dataset/Kinetics400.q_v/40/"
    # "/mnt/HDD10TB-2/ohtani/dataset/Kinetics400.q_v/45/"
    # "/mnt/HDD10TB-2/ohtani/dataset/Kinetics400.q_v/50/"
    )
for model_name in ${models[@]};do
    for path in ${data_path[@]};do
        python dataset_imgaug_copy_3.py --path $path --model_name $model_name --q_use;
    done
done

# python dataset_imgaug_copy.py --model_name x3d_m;
# for i in 0 10 20 30 40 50 60 70 80 90 100; 
#     do
#      python dataset_imgaug_copy.py --compression_rate ${i} --use_compression --model_name x3d_m;
#     done
# done

# for j in `seq 1 10`;
# do
#     python dataset_imgaug.py;
# done
# for i in 0 10 20 30 40 50 60 70 80 90 100; 
#     do
#     for j in `seq 1 10`;
#         do python dataset_imgaug.py --compression_rate ${i} --use_compression;
#     done
# done

# # for i in 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100; 
# # for i in 87 88 89 90 91 92 93 94 95 96 97 98 99 100; 
# models=(
#     "x3d_m"
#     # 'x3d_s'
#     # 'x3d_xs',
#     # 'slow_r50'
#     "x3d_l"
# )
# for model in "${models[@]}";
#     do
#     python dataset_imgaug.py --model_name ${model};
#     done
#     for i in 0 10 20 30 40 50 60 70 80 90 100;
#         do
#         python dataset_imgaug.py --compression_rate ${i} --use_compression --model_name ${model};
#         done
#     done
# done

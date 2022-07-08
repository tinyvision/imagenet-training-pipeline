export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

task="${1:-"quant"}"
struct_txt="${2:-"mixed7d0G"}"
batch="${3:-"512"}"
epochs="${4:-"240"}"
warmup_epochs="${5:-"5"}"
lr="${6:-"0.4"}"
ext_cmd="${7:-" "}"


save_dir=../save_models/${task}/imagenet_${epochs}_bs${batch}_low_aug1
mkdir -p ${save_dir}
cp -r ${task}/${struct_txt}.txt ${save_dir}/

python ../train.py --batch-size ${batch} -j 32 --num-epochs ${epochs} --lr ${lr} \
                        --lr-mode cosine --wd 0.000004 --classes 1000 \
                        --model zennas:${task}/${struct_txt}.txt \
                        --warmup-epochs 5 --no-wd --crop-scale 0.2 --label-smoothing --mixup \
                        --dist-url 'tcp://127.0.0.1:8888' --world-size 1 --rank 0 \
                        --input-size 224 \
                        --train-dir /home/zhenhong.szh/Datasets/Imagenet/train \
                        --val-dir /home/zhenhong.szh/Datasets/Imagenet/valid \
                        --save-dir ${save_dir} \
                        --logging-file ${save_dir}/train.log \
                        --multiprocessing-distributed ${ext_cmd}
                        # --label-smoothing --mixup --random-erasing 0.5 \
pdsh -R ssh -w h[0-15] "python3.5 train_LocalSGD.py --warmup True \
                        --NIID --lr 0.1 --bs 128 --cp 2 --alpha 0.6 --gmf 0.7 \
                        --save -p --name OLocalSGD_nccl_e300_ICASSP_niid \
                        --rank %n --size 16 --backend nccl \
                        --schedule 150 0.1 250 0.1 --epoch 300"


nohup python ./tools/train_val_LUPVisQ.py --dataset ava_database --train_sample_num 25 --val_sample_num 1 --batch_size 64 --num_workers 20 --repeat_num 500 --backbone_type inceptionv3_torchmodel --lambda_ 0.001 >& log.txt&
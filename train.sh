# nohup python ./tools/train_test_objective.py --dataset ava --batch_size 96 --num_workers 20 --model_type objective >& log.txt&
# nohup python ./tools/train_test_subjective.py --dataset ava_database --batch_size 48 --num_workers 20 --model_type subjective >& log.txt&
nohup python ./tools/train_test_LUPVisQ.py --dataset ava_database --batch_size 48 --num_workers 20 --model_type LUPVisQ >& log.txt&
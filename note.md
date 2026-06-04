jupyter nbconvert --to script finetune_inference_grid_experiments.ipynb

nohup python finetune_inference_grid_experiments.py > training.log 2>&1 &

tail -f training.log

pkill -9 python

pid 133433
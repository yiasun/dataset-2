@echo off
python run_t3.py --model mask2former --data-root data\t3 --img-root data\t3\images --work-dir outputs\mask2former --epochs 12 --batch-size 2 --num-classes 80

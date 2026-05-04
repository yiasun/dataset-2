@echo off
python run_t3.py --model cascade_mask_rcnn --data-root data\t3 --img-root data\t3\images --work-dir outputs\cascade_mask_rcnn --epochs 12 --batch-size 2 --num-classes 80

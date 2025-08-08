#!/bin/bash


################# STEP 1 ###################
# unzip sinmdata
if [ -f ../dataset/simdata.zip ]; then
    unzip -o ../dataset/simdata.zip -d ./dataset/
fi

################# STEP 2 ###################
# run training with demo data in count matrix format
python ../train.py --config ../config.JSON \
                --dataset ../dataset \
                --epoch 100 \
                --mark demo_run \
                --format count\ matrix

################# STEP 3 ###################
# run evaluation with demo data in count matrix format
python ../evaluate.py --config /./config.JSON \
                   --dataset .//dataset \
                   --model_path .//output/demo_run_100.pth \
                   --mark demo_evaluation \
                   --format count\ matrix
                   
################# STEP 2 ###################
# run training with demo data in h5ad format
# python train.py --config ../config.JSON \
#                 --dataset ./dataset/data_drop40_716.h5ad \
#                 --epoch 100 \
#                 --mark demo_run \
#                 --format h5ad

################# STEP 3 ###################
# run evaluation with demo data in h5ad format
# python evaluate.py --config ./config.JSON \
#                    --dataset ./dataset/data_drop40_716.h5ad \
#                    --model_path ./output/demo_run_100.pth \
#                    --mark demo_evaluation_h5ad \
#                    --format h5ad


echo "Demo run finished. Check output folder for results."

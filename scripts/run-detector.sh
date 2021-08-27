# !/bin/bash
# run this script from project's dir

export WIDTH=16
export DEPTH=6
export LAYER=6
export N_HID=400
export BATCH_SIZE=128,32
export N_REPEAT=1
export N_BATCH=32
export MAX_EPOCHS=5
export PATIENCE=64
export CONV='het-emb'   # model convolution layer type, choices ['', 'logi', 'gcn', 'gat', 'hgt', 'het-emb']

export PATH_G='./data/g_publish.parquet'
export PATH_DB='./data/feat_store_publish.db'

export DIR_RT="$(pwd)"
echo "Project Dir ${DIR_RT}"

PYTHONPATH="${DIR_RT}:${PYTHONPATH}" \
python xfraud/run_detector.py ${PATH_G} \
    --path-result='exp_result.csv' \
    --width=${WIDTH} --depth=${DEPTH} --batch-size=${BATCH_SIZE} \
    --n-layers=${LAYER} --n-hid=${N_HID}\
    --max-epochs=${MAX_EPOCHS} --patience=${PATIENCE} \
    --n-batch=${N_BATCH} \
    --seed=3030 \
    --path_feat_db=${PATH_DB} \
    --conv-name=${CONV} \
    --seed-epoch
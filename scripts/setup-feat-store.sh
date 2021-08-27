# !/bin/bash
# run this script from project's dir

export DIR_RT="$(pwd)"
echo "Project Dir ${DIR_RT}"

PYTHONPATH="${DIR_RT}:${PYTHONPATH}" \
python xfraud/setup_feature_store.py

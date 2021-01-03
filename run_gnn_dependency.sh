###############################
##### Span baseline
###############################
ROOT_DIR="results/dependency_gcn"
CONFIG="configs/ssqa_dependency_lazy.jsonnet"
PACKAGE="libs"

BATCH_SIZE=8
TARGET_DIR="${ROOT_DIR}/batch_${BATCH_SIZE}-1"
TEST_SET="data/ssqa_multiple_choice_with_dependency/test.json"

python -m allennlp train \
    $CONFIG \
    --serialization-dir $TARGET_DIR \
    --include-package $PACKAGE \
    --overrides "{'data_loader.batch_size':${BATCH_SIZE}}" \
    -f

python -m allennlp predict \
    ${TARGET_DIR}/model.tar.gz \
    $TEST_SET \
    --include-package $PACKAGE \
    --output-file ${TARGET_DIR}/predictions.jsonl \
    --use-dataset-reader \
    --predictor ssqa

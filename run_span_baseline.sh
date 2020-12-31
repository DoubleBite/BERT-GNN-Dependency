###############################
##### Span baseline
###############################
ROOT_DIR="results/span_baseline"
CONFIG="configs/ssqa_span.jsonnet"
PACKAGE="libs"

BATCH_SIZE=16
TARGET_DIR="${ROOT_DIR}/batch_${BATCH_SIZE}-1"
TEST_SET="data/ssqa_multiple_choice_span/test.json"

python -m allennlp train \
    $CONFIG \
    --serialization-dir $TARGET_DIR \
    --include-package $PACKAGE \
    --overrides "{'data_loader.batch_sampler.batch_size':${BATCH_SIZE}}" \
    -f

python -m allennlp predict \
    ${TARGET_DIR}/model.tar.gz \
    $TEST_SET \
    --include-package $PACKAGE \
    --output-file ${TARGET_DIR}/predictions.jsonl \
    --use-dataset-reader \
    --predictor ssqa

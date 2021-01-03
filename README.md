# BERT and GNN for Social Studies Question Answering

BERT and transformers have been a fairly popular choice for question answering (QA) tasks. They are trained to extract answer spans from the passage and have been proven to be effective for QA tasks across various domains. However, for Chinese corpus in which there is no word boundaries, BERT sometimes extracts incomplete answer spans. This phenomenon is even more clear for the social studies domain, where answers are usually in the form of a complex compound noun or a set of nouns. For example: 
1. When the answer is a **compound noun**, BERT usually returns part of it:
    + Question:
        + 「建研所」 是什麼組織的縮寫? (ABRI is the abbreviation for what organization?)
    + Expected answer:
        + 內政部建築研究所 (Architecture and Building Research Institute, Ministry of the Interior)
    + Prediction by BERT:
        + 內政部 (Ministry of the Interior)

2. When the answer is **a set of nouns**, BERT usually returns only one/some of the nouns:
    + Question: 
        + 「阿拉伯之春」運動中，發揮影響力的是那些社群媒體?  
        (Which social media platforms have a large influence on the Arab Spring Revolution?)
    + Expected answer: 
        + 臉書、推特、Youtube (Facebook, Twitter, and Youtube) 
    + Prediction by BERT: 
        + Youtube (Youtube)

In this project, we aim to address the above problems by introducing the dependency information between words. The dependency relationship provides clues to glue relevant words together so that BERT can find more proper answer spans. For example, "內政部" has a dependency relationship "compound:nn" with "建築研究所", so they might be considered together after we add the dependency features. We employ GNN encoders to integrate this dependency information and the resulted `graph representations` are concatenated to the original `BERT representations` before passing into the classifier (as shown below).


<img src="https://i.imgur.com/5t8ibLd.png" width="500">

Specifically, we use `StanfordCoreNLP` and `Stanza` to do dependency parsing and generate the dependency graphs for our corpus. In our model, each `passage-question pair` goes through two types of encoders: the **BERT encoder** to get the normal `BERT representations`, and the **GNN encoder** (GCN or GAT) to get `graph representations` that contains dependency information. Then, two types of representations are concatenated together and passed into the final classifier to calculate the start and end positions of the answer spans. 

We also tried another architecture in our experiment. In addition to the connections (dependency relationships) within the passage and question, we add interconncections across the passage and question so that information can be shared between them. We call this architecture `Dual GNN`. Yet, in our experiment, this change offers minor contribution. Overall, adding a GNN encoder provides 3% performance gain over the baseline BERT QA model.


<img src="https://i.imgur.com/AUVisvR.png" width="500">


## Experiments

### BERT Baseline

We make our baseline BERT-based extractive QA model using AllenNLP.

+ experiment_config: [configs/ssqa_span.jsonnet](configs/ssqa_span.jsonnet).
+ dataset_reader: [libs/dataset_readers/ssqa_span_reader.py](libs/dataset_readers/ssqa_span_reader.py). 
+ model: see [allennlp-models/rc/models/transformer_qa.py](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/rc/models/transformer_qa.py)
+ predictor: [libs/predictors/ssqa_predictor.py](libs/predictors/ssqa_predictor.py).


```bash
python -m allennlp train \
    "configs/ssqa_span.jsonnet" \
    --serialization-dir "results/tmp" \
    --include-package "libs" \
    --overrides "{'data_loader.batch_sampler.batch_size':16}" \
    -f

python -m allennlp predict \
    "results/tmp/model.tar.gz" \
    "data/ssqa_multiple_choice_span/test.json" \
    --include-package "libs" \
    --output-file "results/tmp/predictions.jsonl" \
    --use-dataset-reader \
    --predictor ssqa


# Modify the detailed configuration and run the entire experiment using this shell script
bash run_span_baseline.sh
```


### BERT_GNN
We experiment on different gnn encoders to test whether dependency information helps BERT identify more accurate span borders.

+ experiment_config: [configs/ssqa_dependency_lazy.jsonnet](configs/ssqa_dependency_lazy.jsonnet).
+ dataset_reader: [libs/dataset_readers/ssqa_dependency_reader.py](libs/dataset_readers/ssqa_dependency_reader.py). 
+ model: [libs/models/transformer_gnn.py](libs/models/transformer_gnn.py).
+ modules: [libs/modules/gnn_encoders.py](libs/modules/gnn_encoders.py).
+ predictor: [libs/predictors/ssqa_predictor.py](libs/predictors/ssqa_predictor.py).


```bash
python -m allennlp train \
    "configs/ssqa_dependency_lazy.jsonnet" \
    --serialization-dir "results/tmp2" \
    --include-package "libs" \
    --overrides "{'data_loader.batch_size':16}" \
    -f

python -m allennlp predict \
    "results/tmp2/model.tar.gz" \
    "data/ssqa_multiple_choice_with_dependency/test.json" \
    --include-package "libs" \
    --output-file "results/tmp2/predictions.jsonl" \
    --use-dataset-reader \
    --predictor ssqa

# Modify the detailed configuration and run the entire experiment using this shell script
bash run_gnn_dependency.sh
```
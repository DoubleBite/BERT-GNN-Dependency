# BERT and GNN for Social Studies Question Answering

BERT and transformers have been a fairly popular choice for question answering (QA) tasks. They are trained to extract answer spans from the passage and have been proven to be effective for QA tasks across various domains. However, for Chinese corpus in which there is no word boundaries, BERT sometimes extracts incomplete answer spans. This phenomenon is even more clear for the social studies domain, where answers are usually in the form of a complex compound noun or a set of nouns. For example: 
1. The answer is a **compound noun**:


2. The answer is **a set of nouns**:
    + Question: 「阿拉伯之春」運動中，發揮影響力的是那些社群媒體? <br>
    (Which social media platforms have a large influence on the Arab Spring Revolution?)
    + Expected answer: 臉書、推特、Youtube (Facebook, Twitter, Youtube) 
    + Prediction of BERT: 臉書 (Facebook)

By combining Bert and GNN features, we can integrate both information from the contextualized pretrained model and dependency graph to make more accurate predictions.

To exploit the graph information, however, we meet with the challenge to aggregate information from different graphs, i.e. the passage graph from context passages and the question graph from the question body. To deal with this issue, we also make experiment on different GNN architectures. First, we adopt the framework of (Li et al., 2019), in which the cross-attention mechanism is introduced to propagate information between the passage graph and the question graph. Each node in this architecture is updated with the information of neighboring nodes and nodes from the other graph iteratively. We call this architecture CrossGNN, shown in Figure 16.
We also tried another architecture in which the message propagates within the graphs and between the graphs separately. Hence, it produces two kinds of output features: one is intra-graph features and the other is inter-graph features. We call this architecture DualGNN, as shown in Figure 17.


## Experiments

### BERT Baseline



## Conclusion



Entailment 

不可能会为居民提供什么服务？


居住地的邮局、农会、渔会等组织，提供居民办理借款、存款或提款等服务；此外，邮局还提供邮票的贩售、收寄信件、包裹等服务；农会提供肥料、协助农民作物收购与销售；渔会也协助渔民提升技术，并开发各种特色商品。", "question": "小敏的妈妈目前在邮局服务，请问小敏的妈妈不可能会为居民提供什么服务？", "choices": {"0": "提款、存款", "1": "提供肥料", "2": "收寄信件", "3": "贩售邮票"}, "answer": "提供肥料"




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


<img src="https://i.imgur.com/H1J1yi9.png" width="500">


## Experiments

### BERT Baseline



## Conclusion



Entailment 

不可能会为居民提供什么服务？


居住地的邮局、农会、渔会等组织，提供居民办理借款、存款或提款等服务；此外，邮局还提供邮票的贩售、收寄信件、包裹等服务；农会提供肥料、协助农民作物收购与销售；渔会也协助渔民提升技术，并开发各种特色商品。", "question": "小敏的妈妈目前在邮局服务，请问小敏的妈妈不可能会为居民提供什么服务？", "choices": {"0": "提款、存款", "1": "提供肥料", "2": "收寄信件", "3": "贩售邮票"}, "answer": "提供肥料"




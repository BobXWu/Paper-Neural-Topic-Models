# Papers of Neural Topic Models (NTMs)

[![Awesome](https://awesome.re/badge.svg)](https://github.com/sindresorhus/awesome)
![GitHub stars](https://img.shields.io/github/stars/bobxwu/Paper-Neural-Topic-Models?color=yellow)

## **Check our latest topic modeling toolkit [TopMost](https://github.com/bobxwu/topmost) !**


This repo is a collection of research papers for our survey paper [*A Survey on Neural Topic Models: Methods, Applications, and Challenges*](https://link.springer.com/article/10.1007/s10462-023-10661-7) published at Artificial Intelligence Review.

If you are interested in our survey, please cite as

    @article{wu2023survey,
    title={A Survey on Neural Topic Models: Methods, Applications, and Challenges},
    author={Wu, Xiaobao and Nguyen, Thong and Luu, Anh Tuan},
    journal={Artificial Intelligence Review},
    url={https://doi.org/10.1007/s10462-023-10661-7},
    year={2024},
    publisher={Springer}
    }


- [Papers of Neural Topic Models (NTMs)](#papers-of-neural-topic-models-ntms)
  - [Toolkits](#toolkits)
  - [NTMs with Different Structures](#ntms-with-different-structures)
    - [VAE-based NTM](#vae-based-ntm)
    - [NTMs with Various Priors](#ntms-with-various-priors)
    - [NTMs with Embeddings](#ntms-with-embeddings)
    - [NTMs with External Knowledge](#ntms-with-external-knowledge)
    - [NTMs with PLMs](#ntms-with-plms)
    - [NTMs with Reinforcement Learning](#ntms-with-reinforcement-learning)
    - [NTMs with Adversarial Training](#ntms-with-adversarial-training)
    - [NTMs with Contrastive Learning](#ntms-with-contrastive-learning)
    - [NTMs with Metadata](#ntms-with-metadata)
    - [NTMs with Graphs](#ntms-with-graphs)
    - [Other NTMs](#other-ntms)
  - [NTMs for Various Scenarios](#ntms-for-various-scenarios)
    - [Correlated NTMs](#correlated-ntms)
    - [Short Text NTMs](#short-text-ntms)
    - [Cross-lingual NTMs](#cross-lingual-ntms)
    - [Hierarchical NTMs](#hierarchical-ntms)
    - [Dynamic NTMs](#dynamic-ntms)
    - [Lifelong NTMs](#lifelong-ntms)
  - [Topic Discovery by Clustering](#topic-discovery-by-clustering)
  - [Applications of NTMs](#applications-of-ntms)
    - [Text Analysis](#text-analysis)
    - [Text Generation](#text-generation)
    - [Content Recommendation](#content-recommendation)
  - [Evaluation of Topic Models](#evaluation-of-topic-models)


## Toolkits

1. **OCTIS: Comparing and Optimizing Topic models is Simple!** `EACL 2021` [[pdf]](https://www.aclweb.org/anthology/2021.eacl-demos.31/) [[code]](https://github.com/MIND-Lab/OCTIS)

    *Silvia Terragni, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, Antonio Candelieri*

1. **Towards the TopMost: A Topic Modeling System Toolkit** `2023` [[pdf]](https://arxiv.org/pdf/2309.06908.pdf) [[code]](https://github.com/BobXWu/TopMost)

    *Xiaobao Wu, Fengjun Pan, Anh Tuan Luu*



## NTMs with Different Structures

### VAE-based NTM

1. **Neural Variational Inference for Text Processing** `ICML 2016` [[pdf]](https://arxiv.org/pdf/1511.06038) [[code]](https://github.com/ysmiao/nvdm)

    *Yishu Miao, Lei Yu, Phil Blunsom*

1. **Autoencoding Variational Inference For Topic Models** `ICLR 2017` [[pdf]](https://arxiv.org/pdf/1703.01488) [[code]](https://github.com/akashgit/autoencoding_vi_for_topic_models)

     *Akash Srivastava, Charles Sutton*

1. **Discovering Discrete Latent Topics with Neural Variational Inference** `ICML 2017` [[pdf]](http://proceedings.mlr.press/v70/miao17a/miao17a.pdf)

     *Yishu Miao, Edward Grefenstette, Phil Blunsom*

1. **Coherence-aware Neural Topic Modeling** `EMNLP 2018` [[pdf]](https://arxiv.org/abs/1809.02687) [[code]](https://paperswithcode.com/paper/coherence-aware-neural-topic-modeling)

    *Ran Ding, Ramesh Nallapati, Bing Xiang*

1. **A Discrete Variational Recurrent Topic Model without the Reparametrization Trick** `NeurIPS 2020` [[pdf]](https://proceedings.neurips.cc/paper/2020/file/9f1d5659d5880fb427f6e04ae500fc25-Paper.pdf)

    *Mehdi Rezaee, Francis Ferraro*

1. **Topic Modeling using Variational Auto-Encoders with Gumbel-Softmax and Logistic-Normal Mixture Distributions** `IJCNN 2018` [[pdf]](https://ieeexplore.ieee.org/abstract/document/8489778)

    *Denys Silveira, Andr’e Carvalho, MarcoCristo, Marie-FrancineMoens*

1. **Improving Topic Quality by Promoting Named Entities in Topic Modeling** `ACL 2018` [[pdf]](https://aclanthology.org/P18-2040/)

    *Katsiaryna Krasnashchok, Salim Jouili*

1. **TAN-NTM: Topic Attention Networks for Neural Topic Modeling** `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)

    *Madhur Panwar, Shashank Shailabh, Milan Aggarwal, Balaji Krishnamurthy*


### NTMs with Various Priors

1. **Discovering Discrete Latent Topics with Neural Variational Inference** `ICML 2017` [[pdf]](http://proceedings.mlr.press/v70/miao17a/miao17a.pdf)

    *Yishu Miao, Edward Grefenstette, Phil Blunsom*

1. **Dirichlet Variational Autoencoder** `arXiv 2019` [[pdf]](https://arxiv.org/abs/1901.02739)

    *Weonyoung Joo, Wonsung Lee, Sungrae Park, Il-Chul Moon*

1. **Decoupling Sparsity and Smoothness in the Dirichlet Variational Autoencoder Topic Model** `JMLR 2019` [[pdf]](https://www.jmlr.org/papers/volume20/18-569/18-569.pdf)

    *Sophie Burkhardt, Stefan Kramer*

1. **WHAI: Weibull Hybrid Autoencoding Inference for Deep Topic Modeling** `ICLR 2018` [[pdf]](https://arxiv.org/pdf/1803.01328) [[code]](https://github.com/BoChenGroup/WHAI)

    *Hao Zhang, Bo Chen, Dandan Guo, Mingyuan Zhou*

1. **Learning VAE-LDA Models with Rounded Reparameterization Trick** `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.101/)

    *Runzhi Tian, Yongyi Mao, Richong Zhang*

1. **vONTSS: vMF based semi-supervised neural topic modeling with optimal transport** `EMNLP 2023 Findings` [[pdf]](https://aclanthology.org/2023.findings-acl.271/)  [[code]](https://github.com/xuweijieshuai/Neural-Topic-Modeling-vmf)

    *Weijie Xu, Xiaoyu Jiang, Srinivasan Sengamedu Hanumantha Rao, Francis Iannacci, Jinjin Zhao*




### NTMs with Embeddings

1. **Discovering Discrete Latent Topics with Neural Variational Inference** `ICML 2017` [[pdf]](http://proceedings.mlr.press/v70/miao17a/miao17a.pdf)

    *Yishu Miao, Edward Grefenstette, Phil Blunsom*

1. **Topic Modeling in Embedding Spaces** `TACL 2020` [[pdf]](https://aclanthology.org/2020.tacl-1.29.pdf) [[code]](https://github.com/adjidieng/ETM)

    *Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei*

1. **Neural Topic Model via Optimal Transport** `ICLR 2021` [[pdf]](https://arxiv.org/abs/2008.13537) [[code]](https://github.com/ethanhezhao/NeuralSinkhornTopicModel)

    *He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine*

1. **Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network** `ICML 2021` [[pdf]](http://proceedings.mlr.press/v139/duan21b/duan21b.pdf)

    *Zhibin Duan, Dongsheng Wang, Bo Chen, Chaojie Wang, Wenchao Chen, Yewen Li, Jie Ren, Mingyuan Zhou*

1. **Representing Mixtures of Word Embeddings with Mixtures of Topic Embedding** `ICLR 2022` [[pdf]](https://arxiv.org/abs/2203.01570) [[code]](https://github.com/wds2014/WeTe)

    *Dongsheng Wang, Dandan Guo, He Zhao, Huangjie Zheng, Korawat Tanwisuth, Bo Chen, Mingyuan Zhou*

1. **Bayesian Deep Embedding Topic Meta-Learner** `ICML 2022` [[pdf]](https://proceedings.mlr.press/v162/duan22d/duan22d.pdf)

    *Zhibin Duan, Yishi Xu, Jianqiao Sun, Bo Chen, Wenchao Chen, Chaojie Wang, Mingyuan Zhou*

1. **HyperMiner: Topic Taxonomy Mining with Hyperbolic Embedding** `NeurIPS 2022` [[pdf]](https://arxiv.org/abs/2210.10625) [[code]](https://github.com/NoviceStone/HyperMiner)

    *Yishi Xu, Dongsheng Wang, Bo Chen, Ruiying Lu, Zhibin Duan, Mingyuan Zhou*

1. **Alleviating" Posterior Collapse''in Deep Topic Models via Policy Gradient** `NeurIPS 2022` [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/8d7baf888ca264fd5f2b0d478882b6a2-Paper-Conference.pdf)

    *Yewen Li, Chaojie Wang, Zhibin Duan, Dongsheng Wang, Bo Chen, Bo An, Mingyuan Zhou*

1. **Effective Neural Topic Modeling with Embedding Clustering Regularization** `ICML 2023` [[pdf]](https://arxiv.org/pdf/2306.04217) [[code]](https://github.com/BobXWu/ECRTM)

    *Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu*

### NTMs with External Knowledge

1. **Topicnet: Semantic Graph-Guided Topic Discovery** `NeurIPS 2021` [[pdf]](https://proceedings.neurips.cc/paper/2021/file/0537fb40a68c18da59a35c2bfe1ca554-Paper.pdf)

    *Zhibin Duan, Yishi Xu, Bo Chen, Dongsheng Wang, Chaojie Wang, Mingyuan Zhou*

1. **Knowledge-aware Bayesian Deep Topic Model** `NeurIPS 2022` [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/5c60ee4d6e8faf0f3b2f2701c983dc8c-Paper-Conference.pdf)

    *Dongsheng Wang, Yi Xu, Miaoge Li, Zhibin Duan, Chaojie Wang, Bo Chen, Mingyuan Zhou*

### NTMs with PLMs

1. **Improving Neural Topic Models using Knowledge Distillation** `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.137.pdf) [[code]](https://paperswithcode.com/paper/?acl=2020.emnlp-main.137)

    *Alexander Miserlis Hoyle, Pranav Goel, Philip Resnik*

1. **Cross-Lingual Contextualized Topic Models with Zero-shot Learning** `EACL 2021` [[pdf]](https://aclanthology.org/2021.eacl-main.143.pdf) [[code]](https://paperswithcode.com/paper/?acl=2021.eacl-main.143)

    *Federico Bianchi, Silvia Terragni, Dirk Hovy, Debora Nozza, Elisabetta Fersini*

1. **Large Language Models are Implicitly Topic Models: Explaining and finding good demonstrations for in-context learning** `arXiv 2023` [[pdf]](https://arxiv.org/abs/2301.11916)

    *Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, William Yang Wang*


### NTMs with Reinforcement Learning

1. **Neural Topic Model with Reinforcement Learning** `EMNLP 2019` [[pdf]](https://aclanthology.org/D19-1350/)

    *Lin Gui, Jia Leng, Gabriele Pergola, Yu Zhou, Ruifeng Xu, Yulan He*

1. **Reinforcement Learning for Topic Models** `arXiv 2023` [[pdf]](https://arxiv.org/abs/2012.01524)

    *Jeremy Costello, Marek Z. Reformat*


### NTMs with Adversarial Training

1. **ATM: Adversarial-neural Topic Model** `Information Processing & Management 2019` [[pdf]](https://arxiv.org/pdf/1811.00265)

    *Rui Wang, Deyu Zhou, Yulan He*

1. **Neural Topic Modeling with Bidirectional Adversarial Training** `ACL 2020` [[pdf]](https://aclanthology.org/2020.acl-main.32/) [[code]](https://github.com/zll17/Neural_Topic_Models)

    *Rui Wang, Xuemeng Hu, Deyu Zhou, Yulan He, Yuxuan Xiong, Chenchen Ye, Haiyang Xu*

1. **Neural Topic Modeling with Cycle-consistent Adversarial Training** `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2009.13971)

    *Xuemeng Hu, Rui Wang, Deyu Zhou, Yuxuan Xiong*



### NTMs with Contrastive Learning

1. **Improving Topic Disentanglement via Contrastive Learning** `ACM Information Processing and Management: an International Journal` [[pdf]](https://dl.acm.org/doi/10.1016/j.ipm.2022.103164)

    *Xixi Zhou, Jiajin Bu, Sheng Zhou, Zhi Yu, Ji Zhao, Xifeng Yan*

1. **Contrastive Learning for Neural Topic Model** `NeurIPS 2021` [[pdf]](https://arxiv.org/pdf/2110.12764) [[code]](https://github.com/nguyentthong/CLNTM)

    *Thong Nguyen, Anh Tuan Luu*

1. **Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning** `EMNLP 2022` [[pdf]](https://aclanthology.org/2022.emnlp-main.176) [[code]](https://github.com/bobxwu/TSCTM)

    *Xiaobao Wu, Anh Tuan Luu, Xinshuai Dong*

1. **Topic Modeling as Multi-Objective Contrastive Optimization** `ICLR 2024` [[pdf]](https://arxiv.org/pdf/2402.07577.pdf)

    *Thong Nguyen, Xiaobao Wu, Xinshuai Dong, Cong-Duy Nguyen, See-Kiong Ng, Anh Tuan Luu*

### NTMs with Metadata

1. **Neural Models for Documents with Metadata** `ACL 2018` [[pdf]](https://arxiv.org/pdf/1705.09296) [[code]](https://github.com/dallascard/scholar)

    *Dallas Card, Chenhao Tan, Noah A. Smith*

1. **Discriminative Topic Modeling with Logistic LDA** `arXiv 2019` [[pdf]](https://arxiv.org/abs/1909.01436)

    *Iryna Korshunova, Hanchen Xiong, Mateusz Fedoryszak, Lucas Theis*

1. **Layer-assisted Neural Topic Modeling over Document Networks** `IJCAI 2021` [[pdf]](https://www.ijcai.org/proceedings/2021/0433)

    *Yiming Wang, Ximing Li, Jihong Ouyang*

1. **Neural Topic Model with Attention for Supervised Learning** `AISTATS 2020` [[pdf]](https://proceedings.mlr.press/v108/wang20c.html)

    *Xinyi Wang, Yi Yang*


### NTMs with Graphs

1. **GraphBTM: Graph Enhanced Autoencoded Variational Inference for Biterm Topic Model** `EMNLP 2018` [[pdf]](https://aclanthology.org/D18-1495.pdf) [[code]](https://github.com/valdersoul/GraphBTM)

    *Qile Zhu, Zheng Feng, Xiaolin Li*

1. **Variational Graph Author Topic Modeling** `KDD 2022` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539310?casa_token=mPNGXYJm5hwAAAAA:8J4aSzN7dXmFaT98f13LVh4oF4p1mKm4UZJ_jAQPgcgyfXDOs9YEGpR6Zz_X-eK6LOWcbRCJ0Vdjf2M)

    *Delvin Ce Zhang, Hady W. Lauw*

1. **Topic Modeling on Document Networks with Adjacent-Encoder** `AAAI 2022` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6152/6008) [[code]](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcezhang01%2FAdjacent-Encoder&sa=D&sntz=1&usg=AOvVaw3jGZDHrfjxic8x3teUK0fh)

    *Delvin Ce Zhang, Hady W. Lauw*

1. **Neural Topic Modeling by Incorporating Document Relationship Graph** `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.310/) 

    *Deyu Zhou, Xuemeng Hu, Rui Wang*

1. **Graph Attention Topic Modeling Network** `WWW 2020` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3366423.3380102?casa_token=8fqM-QD88WUAAAAA:E3uNrpuXWVC_4Kd1nZ-fpSrd3_mxClzEx_FY23lsqaHLDryXdsK3NINRPSk4BATi7jJZqSJHP5ewjg)

    *Liang Yang, Fan Wu, Junhua Gu, Chuan Wang, Xiaochun Cao, Di Jin, Yuanfang Guo*

1. **Graph Neural Topic Model with Commonsense Knowledge** `Information Processing & Management 2023` [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0306457322003168)

    *Bingshan Zhu, Yi Cai, Haopeng Ren*

1. **Topic Modeling on Document Networks with Dirichlet Optimal Transport Barycenter** `TKDE 2023` [[pdf]](https://drive.google.com/file/d/1ID-uSY5qvP2tyGRnZ0gdQ01S4TNe2WpJ/view?usp=sharing)

    *Delvin Ce Zhang, Hady W. Lauw*

1. **Hyperbolic Graph Topic Modeling Network with Continuously Updated Topic Tree** `KDD 2023` [[pdf]](https://drive.google.com/file/d/1DWlc7nQ9h1O0LwaRrjelIKk6ScUWIhEm/view?usp=sharing)

    *Delvin Ce Zhang, Rex Ying, and Hady W. Lauw*

1. **ConvNTM: Conversational Neural Topic Model** `AAAI 2023` [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/26595/26367)]

    *Hongda Sun, Quan Tu, Jinpeng Li, Rui Yan*

### Other NTMs

1. **A Neural Autoregressive Topic Model** `NeurIPS 2012` [[pdf]](https://papers.nips.cc/paper/2012/hash/b495ce63ede0f4efc9eec62cb947c162-Abstract.html) 

    *Hugo Larochelle, Stanislas Lauly*

1. **A Novel Neural Topic Model and its Supervised Extension** `AAAI 2015` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/9499/9358)

    *Ziqiang Cao, Sujian Li, Yang Liu, Wenjie Li, Heng Ji*

1. **Document Informed Neural Autoregressive Topic Models with Distributional Prior** `AAAI 2019` [[pdf]](https://arxiv.org/abs/1809.06709)

    *Pankaj Gupta, Yatin Chaudhary, Florian Buettner, Hinrich Schütze*

1. **TextTOvec: Deep Contextualized Neural Autoregressive Topic Models of Language with Distributed Compositional Prior** ` ICLR 2019` [[pdf]](https://arxiv.org/abs/1810.03947)

    *Pankaj Gupta, Yatin Chaudhary, Florian Buettner, Hinrich Schütze*

1. **Sparsemax and Relaxed Wasserstein for Topic Sparsity** `WSDM 2019` [[pdf]](https://arxiv.org/abs/1810.09079)

    *Tianyi Lin, Zhiyue Hu, Xin Guo*

1. **Topic Modeling with Wasserstein Autoencoders** `ACL 2019` [[pdf]](https://aclanthology.org/P19-1640.pdf) [[code]](https://paperswithcode.com/paper/?acl=P19-1640)

    *Feng Nan, Ran Ding, Ramesh Nallapati, Bing Xiang*

1. **Discovering Topics in Long-tailed Corpora with Causal Intervention** `ACL 2021 findings` [[pdf]](https://aclanthology.org/2021.findings-acl.15.pdf) [[code]](https://github.com/bobxwu/DecTM)

    *Xiaobao Wu, Chunping Li, Yishu Miao*




## NTMs for Various Scenarios

### Correlated NTMs

1. **Neural Variational Correlated Topic Modeling** `WWW 2019` [[pdf]](https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313561)

    *Luyang Liu, Heyan Huang, Yang Gao, Yongfeng Zhang, Xiaochi Wei*



### Short Text NTMs

1. **Copula Guided Neural Topic Modelling for Short Texts** `SIGIR 2020` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401245) [[code]](https://github.com/linkstrife/CR-GSM-NVCTM)

    *Lihui Lin, Hongyu Jiang, Yanghui Rao*

1. **Context Reinforced Neural Topic Modeling over Short Texts** `arXiv 2020` [[pdf]](https://arxiv.org/abs/2008.04545)

    *Jiachun Feng, Zusheng Zhang, Cheng Ding, Yanghui Rao, Haoran Xie*

1. **Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder** `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.138.pdf) [[code]](https://github.com/bobxwu/NQTM)

    *Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao*

1. **Extracting Topics with Simultaneous Word Co-occurrence and Semantic Correlation Graphs: Neural Topic Modeling for Short Texts** `EMNLP 2021 findings` [[pdf]](https://aclanthology.org/2021.findings-emnlp.2.pdf)

    *Yiming Wang, Ximing Li, Xiaotang Zhou, Jihong Ouyang*

1. **A Neural Topic Model with Word Vectors and Entity Vectors for Short Texts** `Information Processing & Management 2021` [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S030645732030947X)

    *Xiaowei Zhao, Deqing Wang, Zhengyang Zhao, Wei Liu, Chenwei Lu, Fuzhen Zhuang*

1. **Meta-Complementing the Semantics of Short Texts in Neural Topic Models** `NeurIPS 2022` [[pdf]](https://drive.google.com/file/d/1riuC4cdMNy_0QOhW_Ulik0X6PDcPvXo6/view?usp=sharing)

    *Delvin Ce Zhang, Hady W. Lauw*

1. **Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning** `EMNLP 2022` [[pdf]](https://aclanthology.org/2022.emnlp-main.176) [[code]](https://github.com/bobxwu/TSCTM)

    *Xiaobao Wu, Anh Tuan Luu, Xinshuai Dong*



### Cross-lingual NTMs

1. **Learning Multilingual Topics with Neural Variational Inference** `NLPCC 2020` [[pdf]](https://bobxwu.github.io/files/pub/NLPCC2020_Neural_Multilingual_Topic_Model.pdf) [[code]](https://github.com/BobXWu/NMTM)

    *Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao*


1. **Fine-tuning Encoders for Improved Monolingual and Zero-shot Polylingual Neural Topic Modeling** `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)

    *Aaron Mueller, Mark Dredze*


1. **Multilingual and Multimodal Topic Modelling with Pretrained Embeddings** `COLING 2022`  [[pdf]](https://researchportal.helsinki.fi/files/228080474/COLING_2022_M3L_Topic_Modelling.pdf) [[code]](https://github.com/ezosa/M3L-topic-model)

    *Elaine Zosa, Lidia Pivovarova*


1. **InfoCTM: A Mutual Information Maximization Perspective of Cross-lingual Topic Modeling** `AAAI 2023` [[pdf]](https://arxiv.org/abs/2304.03544) [[code]](https://github.com/bobxwu/infoctm)

    *Xiaobao Wu, Xinshuai Dong, Thong Nguyen, Chaoqun Liu, Liangming Pan, Anh Tuan Luu*




### Hierarchical NTMs

1. **Tree-structured Neural Topic Model** `ACL 2020` [[pdf]](https://aclanthology.org/2020.acl-main.73.pdf)

    *Masaru Isonuma, Junichiro Mori, Danushka Bollegala, Ichiro Sakata*

1. **Tree-Structured Topic Modeling with Nonparametric Neural Variational Inference** `ACL 2021` [[pdf]](https://aclanthology.org/2021.acl-long.182/)

    *Ziye Chen, Cheng Ding, Zusheng Zhang, Yanghui Rao, Haoran Xie*

1. **Neural Attention-aware Hierarchical Topic Model** `arXiv 2021` [[pdf]](https://arxiv.org/abs/2110.07161)

    *Yuan Jin, He Zhao, Ming Liu, Lan Du, Wray Buntine*

1. **Neural Topic Models for Hierarchical Topic Detection and Visualization** `ECML PKDD 2021` [[pdf]](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_219.pdf)

    *Dang Pham, Tuan M. V. Le*

1. **Hierarchical Neural Topic Modeling with Manifold Regularization** `World Wide Web 2021` [[pdf]](https://eprints.usq.edu.au/44204/13/WWWJ-2021.pdf) [[code]](https://github.com/hostnlp/HNTM)

    *Ziye Chen, Cheng Ding, Yanghui Rao, Haoran Xie, Xiaohui Tao, Gary Cheng, Fu Lee Wang*

1. **Bayesian Progressive Deep Topic Model with Knowledge Informed Textual Data Coarsening Process** `ICML 2023` [[pdf]](https://proceedings.mlr.press/v202/duan23c/duan23c.pdf)

    *Zhibin Duan, Xinyang Liu, Yudi Su, Yishi Xu, Bo Chen, Mingyuan Zhou*

1. **Nonlinear Structural Equation Model Guided Gaussian Mixture Hierarchical Topic Modeling** `ACL 2023` [[pdf]](https://aclanthology.org/2023.acl-long.578.pdf) [[code]](https://github.com/nbnbhwyy/NSEM-GMHTM)

    *Hegang Chen, Pengbo Mao, Yuyin Lu, Yanghui Rao*

1. **On the Affinity, Rationality, and Diversity of Hierarchical Topic Modeling** `AAAI 2024` [[pdf]](https://arxiv.org/pdf/2401.14113.pdf) [[code]](https://github.com/BobXWu/TraCo)

    *Xiaobao Wu, Fengjun Pan, Thong Nguyen, Yichao Feng, Chaoqun Liu, Cong-Duy Nguyen, Anh Tuan Luu*


### Dynamic NTMs

1. **The Dynamic Embedded Topic Model** `arXiv 2019` [[pdf]](https://arxiv.org/abs/2012.01524) [[code]](https://github.com/adjidieng/DETM)

    *Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei*

1. **Dynamic Topic Models for Temporal Document Networks** `ICML 2022` [[pdf]](https://proceedings.mlr.press/v162/zhang22n/zhang22n.pdf)

    *Delvin Ce Zhang, Hady W. Lauw*

1. **Neural Dynamic Focused Topic Model** `AAAI 2023` [[pdf]](https://arxiv.org/abs/2012.01524)

    *Kostadin Cvejoski, Ramsés J. Sánchez, César Ojeda*

1. **ANTM: An Aligned Neural Topic Model for Exploring Evolving Topics** `arXiv 2023` [[pdf]](https://arxiv.org/abs/2302.01501)

    *Hamed Rahimi, Hubert Naacke, Camelia Constantin, Bernd Amann*



### Lifelong NTMs

1. **Neural Topic Modeling with Continual Lifelong Learning** `ICML 2020` [[pdf]](https://arxiv.org/abs/2006.10909)

    *Pankaj Gupta, Yatin Chaudhary, Thomas Runkler, Hinrich Schütze*

1. **Lifelong Topic Modeling with Knowledge-enhanced Adversarial Network** `WWW 2022` [[pdf]](https://link.springer.com/article/10.1007/s11280-021-00984-2)

    *Xuewen Zhang, Yanghui Rao, Qing Li*



## Topic Discovery by Clustering

Note that these studies are not real topic models since they can only produce topics while cannot infer the topic distributions of documents as required.


1. **Topic Modeling with Contextualized Word Representation Clusters** `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2010.12626)

    *Laure Thompson and David Mimno*

1. **Top2vec: Distributed Representations of Topics** `arXiv 2020` [[pdf]]([https://arxiv.org/abs/2012.01524](https://arxiv.org/abs/2008.09470))

    *Dimo Angelov*

1. **Topic Modeling with Contextualized Word Representation Clusters** `ACL 2020` [[pdf]](https://arxiv.org/abs/2010.12626)

    *Laure Thompson, David Mimno*

1. **Tired of Topic Models? Clusters of Pretrained Word Embeddings Make for Fast and Good Topics too** `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.135/)

    *Suzanna Sia, Ayush Dalmia, Sabrina J. Mielke*

1. **Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence** `ACL 2021` [[pdf]](https://aclanthology.org/2021.acl-short.96.pdf) [[code]](https://paperswithcode.com/paper/?acl=2021.acl-short.96)

    *Federico Bianchi, Silvia Terragni, Dirk Hovy*

1. **BERTopic: Neural Topic Modeling with a Class-based TF-IDF Procedure** `arXiv 2022` [[pdf]](https://arxiv.org/abs/2203.05794) [[code]](https://github.com/MaartenGr/BERTopic)

    *Maarten Grootendorst*

1. **Is Neural Topic Modelling Better than Clustering? an empirical study on clustering with contextual embeddings for topics** `NAACL 2022` [[pdf]](https://aclanthology.org/2022.naacl-main.285/)

    *Zihan Zhang, Meng Fang, Ling Chen, Mohammad-Reza Namazi-Rad*

1. **Effective Seed-Guided Topic Discovery by Integrating Multiple Types of Contexts** `WSDM 2023` [[pdf]](https://arxiv.org/abs/2212.06002)

    *Yu Zhang, Yunyi Zhang, Martin Michalski, Yucheng Jiang, Yu Meng, Jiawei Han*



## Applications of NTMs

### Text Analysis

1. **Topic Memory Networks for Short Text Classification** `EMNLP 2018` [[pdf]](https://aclanthology.org/D18-1351.pdf) [[code]](https://github.com/zengjichuan/TMN)

    *Jichuan Zeng, Jing Li, Yan Song, Cuiyun Gao, Michael R. Lyu, Irwin King*

1. **Neural Relational Topic Models for Scientific Article Analysis** `CIKM 2018` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3269206.3271696?casa_token=Ak1DPOATCdQAAAAA:5i7CAa9x2dA8XUoN6ZBBLvKinqfR9OsqskT5ZlyJVNQ2vJvfv73q7eIeMqEvxrViP36PtohbB70wtg)

    *Haoli Bai, Zhuangbin Chen, Michael R. Lyu, Irwin King, Zenglin Xu*

1. **Interaction-aware Topic Model for Microblog Conversations through Network Embedding and User Attention** `COLING 2018` [[pdf]](https://aclanthology.org/C18-1118/) 

    *Ruifang He, Xuefei Zhang, Di Jin, Longbiao Wang, Jianwu Dang, Xiangang Li*

1. **TopicBERT for Energy Efficient Document Classification** `EMNLP 2020 Findings` [[pdf]](https://arxiv.org/abs/2010.16407) [[code]](https://github.com/YatinChaudhary/TopicBERT)

    *Yatin Chaudhary, Pankaj Gupta, Khushbu Saxena, Vivek Kulkarni, Thomas Runkler, Hinrich Schütze*

1. **Multi Task Mutual Learning for Joint Sentiment Classification and Topic Detection** `IEEE TKDE 2020` [[pdf]](https://ieeexplore.ieee.org/document/9112648)

    *Lin Gui; Jia Leng; Jiyun Zhou; Ruifeng Xu; Yulan He*

1. **Classification Aware Neural Topic Model for Covid-19 Disinformation Categorisation** `PLOS 2021` [[pdf]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247086)

    *Xingyi Song, Johann Petrak, Ye Jiang, Iknoor Singh, Diana Maynard, Kalina Bontcheva*

1. **Topic Modeling on Podcast Short-text Metadata** `ECIR 2022` [[pdf]](https://arxiv.org/abs/2201.04419)

    *Francisco B. Valero, Marion Baranes, Elena V. Epure*

1. **Topic Modeling Techniques for Text Mining over a Large-Scale Scientific and Biomedical Text Corpus** `IJACI 2022` [[pdf]](https://www.igi-global.com/article/topic-modeling-techniques-for-text-mining-over-a-large-scale-scientific-and-biomedical-text-corpus/293137)

    *Sandhya Avasthi, Ritu Chauhan, Debi Prasanna Acharjya*

1. **Topic Modeling for Multi-Aspect Listwise Comparisons** `CIKM 2021` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3459637.3482398?casa_token=iLrtUseO1P8AAAAA:RgA_5uYNsLR2WSxVYa_6VSSI6ZxJitmBlgcmVSLGznWa_auqE3IHP8S5zO-nWM6L5r3OKQ81N8Ss6Xg) [[code]](https://github.com/cezhang01/malic)

    *Delvin Ce Zhang, Hady W. Lauw*





### Text Generation

1. **A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features** `EMNLP 2019` [[pdf]](https://aclanthology.org/D19-1513/)

    *Hongyin Tang, Miao Li, Beihong Jin*

1. **Topic-Guided Abstractive Text Summarization: a Joint Learning Approach** `arXiv 2020` [[pdf]](https://arxiv.org/abs/2010.10323)

    *Chujie Zheng, Kunpeng Zhang, Harry Jiannan Wang, Ling Fan, Zhe Wang*

1. **What You Say and How You Say it: Joint Modeling of Topics and Discourse in Microblog Conversations** `TACL 2019` [[pdf]](https://arxiv.org/abs/1903.07319)

    *Jichuan Zeng, Jing Li, Yulan He, Cuiyun Gao, Michael R. Lyu, Irwin King*

1. **Enriching and Controlling Global Semantics for Text Summarization** `EMNLP 2021` [[pdf]](https://aclanthology.org/2021.emnlp-main.744/)

    *Thong Nguyen, Anh Tuan Luu, Truc Lu, Tho Quan*

1. **Enhancing Extractive Text Summarization with Topic-Aware Graph Neural Networks** `COLING 2020` [[pdf]](https://arxiv.org/abs/2010.06253)

    *Peng Cui, Le Hu, Yuanchao Liu*

1. **Recurrent Hierarchical Topic-guided RNN for Language Generation** `ICML 2020` [[pdf]](https://arxiv.org/abs/1912.10337)

    *Dandan Guo, Bo Chen, Ruiying Lu, Mingyuan Zhou*

1. **TopNet: Learning from Neural Topic Model to Generate Long Stories** `KDD 2021` [[pdf]](https://arxiv.org/abs/2112.07259)

    *Yazheng Yang, Boyuan Pan, Deng Cai, Huan Sun*

1. **HTKG: Deep Keyphrase Generation with Neural Hierarchical Topic Guidance** `SIGIR 2022` [[pdf]](https://dl.acm.org/doi/abs/10.1145/3477495.3531990)

    *Yuxiang Zhang, Tao Jiang, Tianyu Yang, Xiaoli Li, Suge Wang*

1. **TopicRNN: A Recurrent Neural Network with Long-range Semantic Dependency** `ICLR 2017` [[pdf]](https://arxiv.org/abs/1611.01702) [[code]](https://paperswithcode.com/paper/topicrnn-a-recurrent-neural-network-with-long)

    *Adji B. Dieng, Chong Wang, Jianfeng Gao, John Paisley*
   
1. **DeTiME: Diffusion-Enhanced Topic Modeling using Encoder-decoder based LLM** `EMNLP 2023 Finding` [[pdf]](https://aclanthology.org/2023.findings-emnlp.606.pdf) [[code]](https://github.com/amazon-science/text_generation_diffusion_llm_topic)

    *Weijie Xu, Wenxiang Hu, Fanyou Wu, Srinivasan H. Sengamedu*



### Content Recommendation

1. **Structured Neural Topic Models for Reviews** `AISTATS 2019` [[pdf]](https://arxiv.org/abs/1812.05035) 

    *Babak Esmaeili, Hongyi Huang, Byron C. Wallace, Jan-Willem van de Meent*

1. **Graph Neural Collaborative Topic Model for Citation Recommendation** `ACM TOIS 2021` [[pdf]](https://dl.acm.org/doi/10.1145/3473973?sid=SCITRUS)

    *Qianqian Xie, Yutao Zhu, Jimin Huang, Pan Du, Jian-Yun Nie*




## Evaluation of Topic Models

1. **Evaluation Methods for Topic Models** `ICML 2009` [[pdf]](http://dirichlet.net/pdf/wallach09evaluation.pdf)

    *Hanna M. Wallach, Iain Murray, Ruslan Salakhutdinov, David Mimno*

1. **Reading Tea Leaves: How Humans Interpret Topic Models** `NeurIPS 2009` [[pdf]](https://papers.nips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html)

    *Jonathan Chang, Sean Gerrish, Chong Wang, Jordan Boyd-graber, David Blei*

1. **Estimating Likelihoods for Topic Models** `ACML 2009` [[pdf]](https://arxiv.org/abs/2012.01524)

    *Wray Buntine*

1. **Automatic Evaluation of Topic Coherence** `NAACL 2010` [[pdf]](https://aclanthology.org/N10-1012/)

    *David Newman, Jey Han Lau, Karl Grieser, Timothy Baldwin*

1. **Topic Model or Topic Twaddle? Re-evaluating Semantic Interpretability Measures** `NAACL 2021` [[pdf]](https://aclanthology.org/2021.naacl-main.300/)

    *Caitlin Doogan, Wray Buntine*

1. **Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality** `ACL 2014` [[pdf]](https://aclanthology.org/E14-1056/) [[code]](https://github.com/jhlau/topic_interpretability)

    *Jey Han Lau, David Newman, Timothy Baldwin*

1. **Exploring the Space of Topic Coherence Measures** `WSDM 2015` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/2684822.2685324?casa_token=SZCz7HIe8ecAAAAA:w76e2OqcMLJ6lcuTkU050S_QREP8LNm2kAXpV-O47kAT6FW9jpsBwMp-2Vsa_iDxVxpV0LfkoQSZGA) [[code]](https://github.com/dice-group/Palmetto)

    *Michael Röder, Andreas Both, Alexander Hinneburg*

1. **Is Automated Topic Model Evaluation Broken? The Incoherence of Coherence** `NeurIPS 2021` [[pdf]](https://arxiv.org/abs/2107.02173)

    *Alexander Hoyle, Pranav Goel, Denis Peskov, Andrew Hian-Cheong, Jordan Boyd-Graber, Philip Resnik*

1. **Are Neural Topic Models Broken?** `EMNLP 2022` [[pdf]](https://arxiv.org/abs/2210.16162)

    *Alexander Hoyle, Pranav Goel, Rupak Sarkar, Philip Resnik*

1. **Benchmarking Neural Topic Models: An Empirical Study** `ACL 2021` [[pdf]](https://aclanthology.org/2021.findings-acl.382.pdf)

    *Thanh-Nam Doan, Tuan-Anh Hoang*

1. **Re-visiting Automated Topic Model Evaluation with Large Language Models** `arXiv 2023` [[pdf]](https://arxiv.org/abs/2305.12152)

    *Dominik Stammbach, Vilém Zouhar, Alexander Hoyle, Mrinmaya Sachan, Elliott Ash*

1. **Revisiting Automated Topic Model Evaluation with Large Language Models** `EMNLP 2023` [[pdf]](https://aclanthology.org/2023.emnlp-main.581.pdf)

    *Dominik Stammbach, Vilém Zouhar, Alexander Hoyle, Mrinmaya Sachan, Elliott Ash*

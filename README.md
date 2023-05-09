# Papers of Neural Topic Models (NTMs)

- [Papers of Neural Topic Models (NTMs)](#papers-of-neural-topic-models-ntms)
    - [Survey](#survey)
    - [unclassified](#unclassified)
    - [VAE-based NTMs](#vae-based-ntms)
    - [Neural Topic Models with PLMs](#neural-topic-models-with-plms)
    - [Neural Topic Models with Adversarial Training](#neural-topic-models-with-adversarial-training)
    - [NTM with Contrastive Learning](#ntm-with-contrastive-learning)
    - [Neural Topic Models with Graphs](#neural-topic-models-with-graphs)
    - [Hierarchical NTM](#hierarchical-ntm)
    - [Short Text Neural Topic Models](#short-text-neural-topic-models)
    - [Cross-lingual (Multilingual) Neural Topic Models](#cross-lingual-multilingual-neural-topic-models)
    - [Dynamic NTMs](#dynamic-ntms)
    - [NTMs with Clustering](#ntms-with-clustering)
    - [Others](#others)
    - [Applications of NTMs](#applications-of-ntms)
    - [Evaluation of Topic Models](#evaluation-of-topic-models)


### Survey

1. **Topic Modelling Meets Deep Neural Networks: A Survey** *He Zhao, Dinh Phung, Viet Huynh, Yuan Jin, Lan Du, Wray Buntine* `IJCAI 2021` [[pdf]](https://arxiv.org/pdf/2103.00498)


1. **Topic modeling algorithms and applications: A survey** *


### unclassified

1. **Effective Seed-Guided Topic Discovery by Integrating Multiple Types of Contexts**

1. **Tree-Structured Decoding with DOUBLYRECURRENT Neural Networks**

1. **K-competitive autoencoder for text**

1. **Enhancing extractive text summarization with topic-aware graph neural networks**

1. **Topic modeling in embedding spaces**

1. **Structured neural topic models for reviews**

1. **Context reinforced neural topic modeling over short texts**

1. **Neural topic model with reinforcement learning**

1. **Recurrent hierarchical topic-guided RNN for language generation**

1. **Document informed neural autoregressive topic models with distributional prior**



### VAE-based NTMs

1. **Neural Variational Inference for Text Processing** *Yishu Miao, Lei Yu, Phil Blunsom* `ICML 2016` [[pdf]](https://arxiv.org/pdf/1511.06038) [[code]](https://github.com/ysmiao/nvdm)

1. **Autoencoding Variational Inference For Topic Models** *Akash Srivastava, Charles Sutton* `ICLR 2017` [[pdf]](https://arxiv.org/pdf/1703.01488) [[code]](https://github.com/akashgit/autoencoding_vi_for_topic_models)

1. **Discovering discrete latent topics with neural variational inference** *Yishu Miao, Edward Grefenstette, Phil Blunsom* `ICML 2017` [[pdf]](http://proceedings.mlr.press/v70/miao17a/miao17a.pdf)

1. **Neural Models for Documents with Metadata** *Dallas Card, Chenhao Tan, Noah A. Smith* `ACL 2018` [[pdf]](https://arxiv.org/pdf/1705.09296) [[code]](https://github.com/dallascard/scholar)

1. **Topic Modeling with Wasserstein Autoencoders** *Feng Nan, Ran Ding, Ramesh Nallapati, Bing Xiang* `ACL 2019` [[pdf]](https://aclanthology.org/P19-1640.pdf) [[code]](https://paperswithcode.com/paper/?acl=P19-1640)

1. **Neural Variational Correlated Topic Modeling** *Luyang Liu, Heyan Huang, Yang Gao, Yongfeng Zhang, Xiaochi Wei* `WWW 2019` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3308558.3313561)

1. **Topic Modeling in Embedding Spaces** *Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei* `TACL 2020` [[pdf]](https://aclanthology.org/2020.tacl-1.29.pdf) [[code]](https://paperswithcode.com/paper/?acl=2020.tacl-1.29)

1. **Discovering Topics in Long-tailed Corpora with Causal Intervention** *Xiaobao Wu, Chunping Li, Yishu Miao* `ACL 2021 findings` [[pdf]](https://aclanthology.org/2021.findings-acl.15.pdf) [[code]](https://github.com/bobxwu/DecTM)

1. **Contrastive Learning for Neural Topic Model** *Thong Nguyen, Anh Tuan Luu* `NeurIPS 2021` [[pdf]](https://arxiv.org/pdf/2110.12764) [[code]](https://github.com/nguyentthong/CLNTM)

1. **Neural Topic Model via Optimal Transport** *He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine* `NeurIPS 2021` [[pdf]](https://arxiv.org/pdf/2008.13537) [[code]](https://github.com/ethanhezhao/NeuralSinkhornTopicModel)

1. **Decoupling Sparsity and Smoothness in the Dirichlet Variational Autoencoder Topic Model** *Sophie Burkhardt, Stefan Kramer* `JMLR 2019` [[pdf]](https://www.jmlr.org/papers/volume20/18-569/18-569.pdf)

1. **Coherence-aware Neural Topic Modeling** *Ran Ding, Ramesh Nallapati, Bing Xiang* `EMNLP 2018` [[pdf]](https://arxiv.org/abs/1809.02687) [[code]](https://paperswithcode.com/paper/coherence-aware-neural-topic-modeling)

1. **Tree-structured Neural Topic Model** *Masaru Isonuma, Junichiro Mori, Danushka Bollegala, Ichiro Sakata* `ACL 2020` [[pdf]](https://aclanthology.org/2020.acl-main.73.pdf)


1. **Neural Variational Correlated Topic Modeling** *Luyang Liu, Heyan Huang, Yang Gao, Yongfeng Zhang, Xiaochi Wei* `WWW 2019` [[pdf]](https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313561)


1. **Learning VAE-LDA models with Rounded Reparameterization Trick** *Runzhi Tian, Yongyi Mao, Richong Zhang* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.101/)

1. **A Discrete Variational Recurrent Topic Model without the Reparametrization Trick** *Mehdi Rezaee, Francis Ferraro* `NeurIPS 2020` [[pdf]](https://proceedings.neurips.cc/paper/2020/file/9f1d5659d5880fb427f6e04ae500fc25-Paper.pdf)

1. **Topic Modeling using Variational Auto-Encoders with Gumbel-Softmax and Logistic-Normal Mixture Distributions** *Denys Silveira, Andr’e Carvalho, MarcoCristo, Marie-FrancineMoens* `IJCNN 2018` [[pdf]](https://ieeexplore.ieee.org/abstract/document/8489778)

1. **WHAI: Weibull Hybrid Autoencoding Inference for Deep Topic Modeling** *Hao Zhang, Bo Chen, Dandan Guo, Mingyuan Zhou* `ICLR 2018` [[pdf]](https://arxiv.org/pdf/1803.01328) [[code]](https://github.com/BoChenGroup/WHAI)


### Neural Topic Models with PLMs

1. **Improving Neural Topic Models using Knowledge Distillation** *Alexander Miserlis Hoyle, Pranav Goel, Philip Resnik* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.137.pdf) [[code]](https://paperswithcode.com/paper/?acl=2020.emnlp-main.137)


1. **Cross-lingual Contextualized Topic Models with Zero-shot Learning** *Federico Bianchi, Silvia Terragni, Dirk Hovy, Debora Nozza, Elisabetta Fersini* `EACL 2021` [[pdf]](https://aclanthology.org/2021.eacl-main.143.pdf) [[code]](https://paperswithcode.com/paper/?acl=2021.eacl-main.143)

1. **Auto-encoding variational Bayes.** *Diederik P Kingma, Max Welling* `ICLR 2014` [[pdf]](https://arxiv.org/abs/1312.6114)


### Neural Topic Models with Adversarial Training

1. **ATM: Adversarial-neural Topic Model** *Rui Wang, Deyu Zhou, Yulan He* `Information Processing & Management 2019` [[pdf]](https://arxiv.org/pdf/1811.00265)

1. **Neural Topic Modeling with Bidirectional Adversarial Training** *Rui Wang, Xuemeng Hu, Deyu Zhou, Yulan He, Yuxuan Xiong, Chenchen Ye, Haiyang Xu* `ACL 2020` [[pdf]](https://aclanthology.org/2020.acl-main.32/) [[code]](https://github.com/zll17/Neural_Topic_Models)

1. **Neural Topic Modeling with Cycle-consistent Adversarial Training** *Xuemeng Hu, Rui Wang, Deyu Zhou, Yuxuan Xiong* `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2009.13971)


### NTM with Contrastive Learning

1. Improving topic disentanglement via contrastive learning


### Neural Topic Models with Graphs

1. **GraphBTM: Graph Enhanced Autoencoded Variational Inference for Biterm Topic Model** *Qile Zhu, Zheng Feng, Xiaolin Li* `EMNLP 2018` [[pdf]](https://aclanthology.org/D18-1495.pdf) [[code]](https://github.com/valdersoul/GraphBTM)

1. **Variational Graph Author Topic Modeling** *Delvin Ce Zhang, Hady W. Lauw* `KDD 2022` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539310?casa_token=mPNGXYJm5hwAAAAA:8J4aSzN7dXmFaT98f13LVh4oF4p1mKm4UZJ_jAQPgcgyfXDOs9YEGpR6Zz_X-eK6LOWcbRCJ0Vdjf2M)

1. **Topic Modeling on Document Networks with Adjacent-Encoder** *Delvin Ce Zhang, Hady W. Lauw* `AAAI 2022` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6152/6008) [[code]](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcezhang01%2FAdjacent-Encoder&sa=D&sntz=1&usg=AOvVaw3jGZDHrfjxic8x3teUK0fh)

1. **Neural Topic Modeling by Incorporating Document Relationship Graph** *Deyu Zhou, Xuemeng Hu, Rui Wang* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.310/) 

1. **Graph Attention Topic Modeling Network** *Liang Yang, Fan Wu, Junhua Gu, Chuan Wang, Xiaochun Cao, Di Jin, Yuanfang Guo* `WWW 2020` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3366423.3380102?casa_token=8fqM-QD88WUAAAAA:E3uNrpuXWVC_4Kd1nZ-fpSrd3_mxClzEx_FY23lsqaHLDryXdsK3NINRPSk4BATi7jJZqSJHP5ewjg)


### Hierarchical NTM

1. **Neural attention-aware hierarchical topic model**


### Short Text Neural Topic Models

1. **Copula Guided Neural Topic Modelling for Short Texts** *Lihui Lin, Hongyu Jiang, Yanghui Rao* `SIGIR 2020` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401245) [[code]](https://github.com/linkstrife/CR-GSM-NVCTM)

1. **Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder** *Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.138.pdf) [[code]](https://github.com/bobxwu/NQTM)

1. **Extracting Topics with Simultaneous Word Co-occurrence and Semantic Correlation Graphs: Neural Topic Modeling for Short Texts** *Yiming Wang, Ximing Li, Xiaotang Zhou, Jihong Ouyang* `EMNLP 2021 findings` [[pdf]](https://aclanthology.org/2021.findings-emnlp.2.pdf)

1. **Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning** *Xiaobao Wu, Anh Tuan Luu, Xinshuai Dong* `EMNLP 2022` [[pdf]]() [[code]](https://github.com/bobxwu/TSCTM)





### Cross-lingual (Multilingual) Neural Topic Models

1. **Learning Multilingual Topics with Neural Variational Inference** *Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao* `NLPCC 2020` [[pdf]](https://bobxwu.github.io/files/pub/NLPCC2020_Neural_Multilingual_Topic_Model.pdf) [[code]](https://github.com/BobXWu/NMTM)

1. **Multilingual and Multimodal Topic Modelling with Pretrained Embeddings** *Elaine Zosa, Lidia Pivovarova* `COLING 2022`  [[pdf]](https://researchportal.helsinki.fi/files/228080474/COLING_2022_M3L_Topic_Modelling.pdf) [[code]](https://github.com/ezosa/M3L-topic-model)





### Dynamic NTMs

1. **Dynamic Topic Models for Temporal Document Networks** *Delvin Ce Zhang, Hady W. Lauw* `ICML 2022` [[pdf]](https://proceedings.mlr.press/v162/zhang22n/zhang22n.pdf)



### NTMs with Clustering

Note that some studies are not real topic models since they can only produce topics while cannot infer the topic distributions of documents as required.


1. **Topic Modeling with Contextualized Word Representation Clusters** *Laure Thompson and David Mimno* `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2010.12626)


1. **Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence** *Federico Bianchi, Silvia Terragni, Dirk Hovy* `ACL 2021` [[pdf]](https://aclanthology.org/2021.acl-short.96.pdf) [[code]](https://paperswithcode.com/paper/?acl=2021.acl-short.96)


### Others


1. **A Novel Neural Topic Model and its Supervised Extension** * Ziqiang Cao, Sujian Li, Yang Liu, Wenjie Li, Heng Ji* `AAAI 2015` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/9499/9358)

1. **Neural Topic Model with Reinforcement Learning** *Lin Gui, Jia Leng, Gabriele Pergola, Yu Zhou, Ruifeng Xu, Yulan He* `EMNLP 2019` [[pdf]](https://aclanthology.org/D19-1350/) 

1. **A Neural Autoregressive Topic Model** *Hugo Larochelle, Stanislas Lauly* `NeurIPS 2012` [[pdf]](https://papers.nips.cc/paper/2012/hash/b495ce63ede0f4efc9eec62cb947c162-Abstract.html) 





### Applications of NTMs

1. **Topic Memory Networks for Short Text Classification** *Jichuan Zeng, Jing Li, Yan Song, Cuiyun Gao, Michael R. Lyu, Irwin King* `EMNLP 2018` [[pdf]](https://aclanthology.org/D18-1351.pdf) [[code]](https://github.com/zengjichuan/TMN)

1. **Topic Modeling for Multi-Aspect Listwise Comparisons** *Delvin Ce Zhang, Hady W. Lauw* `CIKM 2021` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3459637.3482398?casa_token=iLrtUseO1P8AAAAA:RgA_5uYNsLR2WSxVYa_6VSSI6ZxJitmBlgcmVSLGznWa_auqE3IHP8S5zO-nWM6L5r3OKQ81N8Ss6Xg) [[code]](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcezhang01%2Fmalic&sa=D&sntz=1&usg=AOvVaw2Bqy_hLzRu13FRQdad_y_b)

1. **Neural Relational Topic Models for Scientific Article Analysis** *Haoli Bai, Zhuangbin Chen, Michael R. Lyu, Irwin King, Zenglin Xu* `CIKM 2018` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3269206.3271696?casa_token=Ak1DPOATCdQAAAAA:5i7CAa9x2dA8XUoN6ZBBLvKinqfR9OsqskT5ZlyJVNQ2vJvfv73q7eIeMqEvxrViP36PtohbB70wtg)


1. **TopicBERT for Energy Efficient Document Classification** *Yatin Chaudhary, Pankaj Gupta, Khushbu Saxena, Vivek Kulkarni, Thomas Runkler, Hinrich Schütze* `EMNLP 2020 Findings` [[pdf]](https://arxiv.org/abs/2010.16407) [[code]](https://github.com/YatinChaudhary/TopicBERT)

1. **TopicRNN: A Recurrent Neural Network with Long-range Semantic Dependency** *Adji B. Dieng, Chong Wang, Jianfeng Gao, John Paisley* `ICLR 2017` [[pdf]](https://arxiv.org/abs/1611.01702) [[code]](https://paperswithcode.com/paper/topicrnn-a-recurrent-neural-network-with-long)

1. **Structured Neural Topic Models for Reviews** *Babak Esmaeili, Hongyi Huang, Byron C. Wallace, Jan-Willem van de Meent* `AISTATS 2019` [[pdf]](https://arxiv.org/abs/1812.05035) 

1. **Interaction-aware Topic Model for Microblog Conversations through Network Embedding and User Attention** *Ruifang He, Xuefei Zhang, Di Jin, Longbiao Wang, Jianwu Dang, Xiangang Li* `COLING 2018` [[pdf]](https://aclanthology.org/C18-1118/) 



### Evaluation of Topic Models

1. **Evaluation Methods for Topic Models** *Hanna M. Wallach, Iain Murray, Ruslan Salakhutdinov, David Mimno* `ICML 2009` [[pdf]](http://dirichlet.net/pdf/wallach09evaluation.pdf)

1. **Reading tea leaves: How humans interpret topic models** *Jonathan Chang, Sean Gerrish, Chong Wang, Jordan Boyd-graber, David Blei* `NeurIPS 2009` [[pdf]](https://papers.nips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html)

1. **Topic Model or Topic Twaddle? Re-evaluating Semantic Interpretability Measures** *Caitlin Doogan, Wray Buntine* `NAACL 2021` [[pdf]](https://aclanthology.org/2021.naacl-main.300/)

1. **Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality** *Jey Han Lau, David Newman, Timothy Baldwin* `ACL 2014` [[pdf]](https://aclanthology.org/E14-1056/) [[code]](https://github.com/jhlau/topic_interpretability)

1. **Exploring the Space of Topic Coherence Measures** *Michael Röder, Andreas Both, Alexander Hinneburg* `WSDM 2015` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/2684822.2685324?casa_token=SZCz7HIe8ecAAAAA:w76e2OqcMLJ6lcuTkU050S_QREP8LNm2kAXpV-O47kAT6FW9jpsBwMp-2Vsa_iDxVxpV0LfkoQSZGA) [[code]](https://github.com/dice-group/Palmetto)

1. **Is automated topic model evaluation broken? the incoherence of coherence**

1. **Are Neural Topic Models Broken?**


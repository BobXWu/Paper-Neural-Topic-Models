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

1. **Effective Seed-Guided Topic Discovery by Integrating Multiple Types of Contexts** *Yu Zhang, Yunyi Zhang, Martin Michalski, Yucheng Jiang, Yu Meng, Jiawei Han* `WSDM 2023` [[pdf]](https://arxiv.org/abs/2212.06002)

1. **Tree-Structured Decoding with DOUBLYRECURRENT Neural Networks** *David Alvarez-Melis, Tommi S. Jaakkola* `ICLR 2017` [[pdf]](https://openreview.net/forum?id=HkYhZDqxg)

1. **K-competitive autoencoder for text** *Yu Chen, Mohammed J. Zaki* `KDD 2017` [[pdf]](https://arxiv.org/abs/1705.02033)

1. **Enhancing extractive text summarization with topic-aware graph neural networks** *Peng Cui, Le Hu, Yuanchao Liu* `COLING 2020` [[pdf]](https://arxiv.org/abs/2010.06253)

1. **Topic modeling in embedding spaces** *Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1907.04907) 

1. **Structured neural topic models for reviews** *Babak Esmaeili, Hongyi Huang, Byron C. Wallace, Jan-Willem van de Meent* `arXiv 2018` [[pdf]](https://arxiv.org/abs/1812.05035)

1. **Context reinforced neural topic modeling over short texts** *Jiachun Feng, Zusheng Zhang, Cheng Ding, Yanghui Rao, Haoran Xie* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2008.04545)

1. **Neural topic model with reinforcement learning** *Lin Gui, Jia Leng, Gabriele Pergola, Yu Zhou, Ruifeng Xu, Yulan He* `EMNLP 2019` [[pdf]](https://aclanthology.org/D19-1350/)

1. **Recurrent hierarchical topic-guided RNN for language generation** *Dandan Guo, Bo Chen, Ruiying Lu, Mingyuan Zhou* `ICML 2020` [[pdf]](https://arxiv.org/abs/1912.10337)

1. **Document informed neural autoregressive topic models with distributional prior** *Pankaj Gupta, Yatin Chaudhary, Florian Buettner, Hinrich Schütze* `AAAI 2019` [[pdf]](https://arxiv.org/abs/1809.06709)

1. **BERTopic: Neural topic modeling with a class-based TF-IDF procedure** *Maarten Grootendorst* `arXiv 2022` [[pdf]](https://arxiv.org/abs/2203.05794)

1. **Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning** *Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, William Yang Wang* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2301.11916)

1. **Topic Modeling With Topological Data Analysis** *Ciarán Byrne, Danijela Horak, Karo Moilanen, Amandla Mabona* `EMNLP 2022` [[pdf]](https://aclanthology.org/2022.emnlp-main.792/)

1. **Tree-Structured Topic Modeling with Nonparametric Neural Variational Inference** *Ziye Chen, Cheng Ding, Zusheng Zhang, Yanghui Rao, Haoran Xie* `ACL 2021` [[pdf]](https://aclanthology.org/2021.acl-long.182/)

1. **TAN-NTM: Topic Attention Networks for Neural Topic Modeling** *Madhur Panwar, Shashank Shailabh, Milan Aggarwal, Balaji Krishnamurthy* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Top2vec: Distributed representations of topics** *Dimo Angelov* `arXiv 2020` [[pdf]]([https://arxiv.org/abs/2012.01524](https://arxiv.org/abs/2008.09470))

1. **Topic modeling techniques for text mining over a large-scale scientific and biomedical text corpus** *Sandhya Avasthi, Ritu Chauhan, Debi Prasanna Acharjya* `IJACI 2022` [[pdf]]([https://arxiv.org/abs/2012.01524](https://www.igi-global.com/article/topic-modeling-techniques-for-text-mining-over-a-large-scale-scientific-and-biomedical-text-corpus/293137))

1. **Correlated topic models** *John Lafferty, David Blei* `NeurIPS 2005` [[pdf]]([https://arxiv.org/abs/2012.01524](https://proceedings.neurips.cc/paper_files/paper/2005/hash/9e82757e9a1c12cb710ad680db11f6f1-Abstract.html))

1. **Probabilistic topic models** *Blei DM* `Communications of the ACM 2012` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Dynamic topic models** *Blei DM, Lafferty JD* `ICML 2006` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Latent dirichlet allocation** *Blei DM, Ng AY, Jordan MI* `JLMR 2003` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **The nested chinese restaurant process and bayesian nonparametric inference of topic hierarchies.** *Blei DM, Griffiths TL, Jordan MI* `JACM 2010` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Variational inference: A review for statisticians** *Blei DM, Kucukelbir A, McAuliffe JD* `Journal of the American Statistical Association 2017` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Applications of topic models** *Boyd-Graber JL, Hu Y, Mimno D, et al*  [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Estimating likelihoods for topic models** *Buntine WL* `ACML 2009` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Reinforcement learning for topic models** *Costello J, Reformat MZ* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Neural dynamic focused topic model** *Cvejoski K, S´anchez RJ, Ojeda C* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **The dynamic embedded topic model** *Dieng AB, Ruiz FJ, Blei DM* `arXiv 2019` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Topic modeling in embedding spaces** *Dieng AB, Ruiz FJ, Blei DM* `TACL 2020` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Benchmarking neural topic models: An empirical study** *Doan TN, Hoang TA* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Hierarchical topic models and the nested chinese restaurant process** *Griffiths T, Jordan M, Tenenbaum J, et al* `NeurIPS 2023` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Neural topic modeling with continual lifelong learning** *Gupta P, Chaudhary Y, Runkler T, et al* `ICML 2020` [[pdf]](https://arxiv.org/abs/2006.10909)

1. **Multi task mutual learning for joint sentiment classification and topic detection** *Gui L, Leng J, Zhou J, et al* `IEEE Transactions on Knowledge and Data Engineering 2020` [[pdf]](https://ieeexplore.ieee.org/document/9112648)

1. **Kernel topic models** *Gupta P, Chaudhary Y, Runkler T, et al* `arXiv 2011` [[pdf]](https://arxiv.org/abs/1110.4713)

1. **Replicated softmax: an undirected topic model** *Hinton GE, Salakhutdinov RR* `NeurIPS 2009` [[pdf]](https://papers.nips.cc/paper_files/paper/2009/hash/31839b036f63806cba3f47b93af8ccb5-Abstract.html)

1. **Dirichlet variational autoencoder** *Joo W, Lee W, Park S, et al* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1901.02739)

1. **Simultaneous discovery of common and discriminative topics via joint nonnegative matrix factorization** *Kim H, Choo J, Kim J, et al* `KDD 2015` [[pdf]](https://dl.acm.org/doi/10.1145/2783258.2783338)

1. **Discriminative topic modeling with logistic lda** *Korshunova I, Xiong H, Fedoryszak M, et al* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1909.01436)

1. **Improving Topic Quality by Promoting Named Entities in Topic Modeling** *Krasnashchok K, Jouili S* `ACL 2018` [[pdf]](https://aclanthology.org/P18-2040/)

1. **A systematic review of the use of topic models for short text social media analysis** *Laureate CDP, Buntine W, Linger H* `Artificial Intelligence Review 2023` [[pdf]](https://link.springer.com/article/10.1007/s10462-023-10471-x)

1. **Sparsemax and relaxed wasserstein for topic sparsity** *Lin T, Hu Z, Guo X* `WSDM 2019` [[pdf]](https://arxiv.org/abs/1810.09079)

1. **Plda+ parallel latent dirichlet allocation with data placement and pipeline processing** *Liu Z, Zhang Y, Chang EY, et al* `ACM Transactions on Intelligent Systems and Technology 2011` [[pdf]](https://dl.acm.org/doi/10.1145/1961189.1961198)

1. **Supervised topic models** *Mcauliffe J, Blei D* `NeurIPS 2007` [[pdf]](https://papers.nips.cc/paper_files/paper/2007/hash/d56b9fc4b0f1be8871f5e1c40c0067e7-Abstract.html)

1. **Neural variational inference for text processing** *Miao Y, Yu L, Blunsom P* `ICML 2016` [[pdf]](https://arxiv.org/abs/1511.06038)

1. **Polylingual topic models** *Mimno D, Wallach H, Naradowsky J, et al* `EMNLP 2009` [[pdf]](https://aclanthology.org/D09-1092/)

1. **Optimizing semantic coherence in topic models** *Mimno D, Wallach HM, Talley E, et al* `EMNLP 2011` [[pdf]](https://dl.acm.org/doi/10.5555/2145432.2145462)

1. **Fine-tuning encoders for improved monolingual and zeroshot polylingual neural topic modeling** *Mueller A, Dredze M* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Distributed algorithms for topic models** *Newman D, Asuncion A, Smyth P, et al* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2104.05064)

1. **Automatic evaluation of topic coherence** *Newman D, Lau JH, Grieser K, et al* `NAACL 2010` [[pdf]](https://aclanthology.org/N10-1012/)

1. **Enriching and controlling global semantics for text summarization** *Nguyen T, Luu AT, Lu T, et al* `EMNLP 2021` [[pdf]](https://aclanthology.org/2021.emnlp-main.744/)

1. **Neural discrete representation learning** *Van den Oord A, Vinyals O* `arXiv 2017` [[pdf]](https://arxiv.org/abs/1711.00937)

1. **Representation learning with contrastive predictive coding** *Van den Oord A, Li Y, Vinyals O* `arXiv 2018` [[pdf]](https://arxiv.org/abs/1807.03748)

1. **Neural topic models for hierarchical topic detection and visualization** *Pham D, Le TM* `ECML PKDD 2021` [[pdf]](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_219.pdf)

1. **Antm: An aligned neural topic model for exploring evolving topics** *Rahimi H, Naacke H, Constantin C, et al* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2302.01501)

1. **Detecting common discussion topics across culture from news reader comments** *Shi B, Lam W, Bing L, et al* `ACL 2016` [[pdf]](https://aclanthology.org/P16-1064/)

1. **Short-text topic modeling via non-negative matrix factorization enriched with local word-context correlations** *Shi T, Kang K, Choo J, et al* `WWW 2018` [[pdf]](https://dl.acm.org/doi/10.1145/3178876.3186009)

1. **Tired of topic models? clusters of pretrained word embeddings make for fast and good topics too** *Sia S, Dalmia A, Mielke SJ* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.135/)

1. **Classification aware neural topic model for covid-19 disinformation categorisation** *Song X, Petrak J, Jiang Y, et al* `PLOS 2021` [[pdf]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247086)

1. **Re-visiting automated topic model evaluation with large language models** *Stammbach D, Zouhar V, Hoyle A, et al* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2305.12152)

1. **Topic modeling with contextualized word representation clusters** *Thompson L, Mimno D* `ACL 2020` [[pdf]](https://arxiv.org/abs/2010.12626)

1. **Topic modeling on podcast short-text metadata** *Valero FB, Baranes M, Epure EV* `ECIR 2022` [[pdf]](https://arxiv.org/abs/2201.04419)

1. **Collaborative topic modeling for recommending scientific articles** *Valero FB, Baranes M, Epure EV* `KDD 2011` [[pdf]](https://dl.acm.org/doi/10.1145/2020408.2020480)

1. **Representing mixtures of word embeddings with mixtures of topic embeddings** *Wang H, He R, Liu H, et al* `ICLR 2022` [[pdf]](https://arxiv.org/abs/2203.01570)

1. **Neural topic model with attention for supervised learning** *Wang X, Yang Y* `AISTATS 2020` [[pdf]](https://proceedings.mlr.press/v108/wang20c.html)

1. **Plda: Parallel latent dirichlet allocation for large-scale applications** *Wang Y, Bai H, Stanton M, et al* `AAIM 2009` [[pdf]](https://link.springer.com/chapter/10.1007/978-3-642-02158-9_26)

1. **Layer-assisted neural topic modeling over document networks** *Wang Y, Li X, Ouyang J* `IJCAI 2021` [[pdf]](https://www.ijcai.org/proceedings/2021/0433)

1. **Short Text Topic Modeling with Flexible Word Patterns** *Wu X, Li C* `IJCNN 2019` [[pdf]](https://ieeexplore.ieee.org/document/8852366)

1. **Infoctm: A mutual information maximization perspective of cross-lingual topic modeling** *Wu X, Dong X, Nguyen T, et al* `AAAI 2023` [[pdf]](https://arxiv.org/abs/2304.03544)

1. **Effective neural topic modeling with embedding clustering regularization** *Wu X, Dong X, Nguyen T, et al* `ICML 2023` 

1. **Graph neural collaborative topic model for citation recommendation** *Xie Q, Zhu Y, Huang J, et al* `ACM Transactions on Information Systems 2021` [[pdf]](https://dl.acm.org/doi/10.1145/3473973?sid=SCITRUS)

1. **Hyperminer: Topic taxonomy mining with hyperbolic embedding** *Xu Y, Wang D, Chen B, et al* `arXiv 2022` [[pdf]](https://arxiv.org/abs/2210.10625)

1. **A biterm topic model for short texts** *Yan X, Guo J, Lan Y, et al* `WWW 2013` [[pdf]](https://dl.acm.org/doi/10.1145/2488388.2488514)

1. **Topnet: Learning from neural topic model to generate long stories** *Yang Y, Pan B, Cai D, et al* `KDD 2021` [[pdf]](https://arxiv.org/abs/2112.07259)

1. **A dirichlet multinomial mixture model-based approach for short text clustering** *Yin J, Wang J* `KDD 2014` [[pdf]](https://dl.acm.org/doi/10.1145/2623330.2623715)

1. **Multilingual anchoring: Interactive topic modeling and alignment across languages** *Yuan M, Van Durme B, Ying JL* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **What you say and how you say it: Joint modeling of topics and discourse in microblog conversations** *Zeng J, Li J, He Y, et al* `TACL 2019` [[pdf]](https://arxiv.org/abs/1903.07319)

1. **Lifelong topic modeling with knowledge-enhanced adversarial network** *Zhang X, Rao Y, Li Q* `WWW 2022` [[pdf]](https://link.springer.com/article/10.1007/s11280-021-00984-2)

1. **Htkg: Deep keyphrase generation with neural hierarchical topic guidance** *Zhang Y, Jiang T, Yang T, et al* `SIGIR 2022` [[pdf]](https://dl.acm.org/doi/abs/10.1145/3477495.3531990)

1. **Is neural topic modelling better than clustering? an empirical study on clustering with contextual embeddings for topics** *Zhang Z, Fang M, Chen L, et al* `NAACL 2022` [[pdf]](https://aclanthology.org/2022.naacl-main.285/)

1. **Topic modelling meets deep neural networks: A survey** *Zhao H, Phung D, Huynh V, et al* `arXiv 2021` [[pdf]](https://arxiv.org/abs/2103.00498)

1. **Neural topic model via optimal transport** *Wang Y, Bai H, Stanton M, et al* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2008.13537)

1. **A neural topic model with word vectors and entity vectors for short texts** *Zhao X, Wang D, Zhao Z, et al* `Information Processing & Management 2021` [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S030645732030947X)

1. **Graph neural topic model with commonsense knowledge** *Zhu B, Cai Y, Ren H* `Information Processing & Management 2023` [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0306457322003168)



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

1. **Improving topic disentanglement via contrastive learning** *Xixi Zhou, Jiajin Bu, Sheng Zhou, Zhi Yu, Ji Zhao, Xifeng Yan* `ACM Information Processing and Management: an International Journal` [[pdf]](https://dl.acm.org/doi/10.1016/j.ipm.2022.103164)


### Neural Topic Models with Graphs

1. **GraphBTM: Graph Enhanced Autoencoded Variational Inference for Biterm Topic Model** *Qile Zhu, Zheng Feng, Xiaolin Li* `EMNLP 2018` [[pdf]](https://aclanthology.org/D18-1495.pdf) [[code]](https://github.com/valdersoul/GraphBTM)

1. **Variational Graph Author Topic Modeling** *Delvin Ce Zhang, Hady W. Lauw* `KDD 2022` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539310?casa_token=mPNGXYJm5hwAAAAA:8J4aSzN7dXmFaT98f13LVh4oF4p1mKm4UZJ_jAQPgcgyfXDOs9YEGpR6Zz_X-eK6LOWcbRCJ0Vdjf2M)

1. **Topic Modeling on Document Networks with Adjacent-Encoder** *Delvin Ce Zhang, Hady W. Lauw* `AAAI 2022` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6152/6008) [[code]](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcezhang01%2FAdjacent-Encoder&sa=D&sntz=1&usg=AOvVaw3jGZDHrfjxic8x3teUK0fh)

1. **Neural Topic Modeling by Incorporating Document Relationship Graph** *Deyu Zhou, Xuemeng Hu, Rui Wang* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.310/) 

1. **Graph Attention Topic Modeling Network** *Liang Yang, Fan Wu, Junhua Gu, Chuan Wang, Xiaochun Cao, Di Jin, Yuanfang Guo* `WWW 2020` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3366423.3380102?casa_token=8fqM-QD88WUAAAAA:E3uNrpuXWVC_4Kd1nZ-fpSrd3_mxClzEx_FY23lsqaHLDryXdsK3NINRPSk4BATi7jJZqSJHP5ewjg)


### Hierarchical NTM

1. **Neural attention-aware hierarchical topic model** *Yuan Jin, He Zhao, Ming Liu, Lan Du, Wray Buntine* `arXiv 2021` [[pdf]](https://arxiv.org/abs/2110.07161)


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

1. **Is automated topic model evaluation broken? the incoherence of coherence** *Alexander Hoyle, Pranav Goel, Denis Peskov, Andrew Hian-Cheong, Jordan Boyd-Graber, Philip Resnik* `NeurIPS 2021` [[pdf]](https://arxiv.org/abs/2107.02173)

1. **Are Neural Topic Models Broken?** *Alexander Hoyle, Pranav Goel, Rupak Sarkar, Philip Resnik* `EMNLP 2022` [[pdf]](https://arxiv.org/abs/2210.16162)


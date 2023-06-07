# Papers of Neural Topic Models (NTMs)

- [Papers of Neural Topic Models (NTMs)](#papers-of-neural-topic-models-ntms)
  - [NTMs with Different Structures](#ntms-with-different-structures)
    - [VAE-based NTMs](#vae-based-ntms)
    - [NTMs with Various Priors](#ntms-with-various-priors)
    - [NTMs with Embeddings](#ntms-with-embeddings)
    - [NTMs with PLMs](#ntms-with-plms)
    - [NTMs with Reinforcement Learning](#ntms-with-reinforcement-learning)
    - [NTMs with Adversarial Training](#ntms-with-adversarial-training)
    - [NTMs with Contrastive Learning](#ntms-with-contrastive-learning)
    - [NTMs with Metadata](#ntms-with-metadata)
    - [NTMs with Graphs](#ntms-with-graphs)
    - [Other NTMs](#other-ntms)
  - [NTMs for Various Scenarios](#ntms-for-various-scenarios)
    - [Correlated NTMs](#correlated-ntms)
    - [Hierarchical NTMs](#hierarchical-ntms)
    - [Short Text NTMs](#short-text-ntms)
    - [Cross-lingual NTMs](#cross-lingual-ntms)
    - [Hierarchical NTMs](#hierarchical-ntms-1)
    - [Dynamic NTMs](#dynamic-ntms)
    - [Lifelong NTMs](#lifelong-ntms)
  - [Discovering Topics by Clustering](#discovering-topics-by-clustering)
  - [Applications of NTMs](#applications-of-ntms)
    - [Text Analysis](#text-analysis)
    - [Text Generation](#text-generation)
    - [Content Recommendation](#content-recommendation)
  - [Evaluation of Topic Models](#evaluation-of-topic-models)


<!-- ## Survey

1. **Topic Modelling Meets Deep Neural Networks: A Survey** *He Zhao, Dinh Phung, Viet Huynh, Yuan Jin, Lan Du, Wray Buntine* `IJCAI 2021` [[pdf]](https://arxiv.org/pdf/2103.00498) -->


<!-- ### unclassified -->

<!-- 1. **Topic Modeling With Topological Data Analysis** *Ciarán Byrne, Danijela Horak, Karo Moilanen, Amandla Mabona* `EMNLP 2022` [[pdf]](https://aclanthology.org/2022.emnlp-main.792/) -->

<!-- 1. **TAN-NTM: Topic Attention Networks for Neural Topic Modeling** *Madhur Panwar, Shashank Shailabh, Milan Aggarwal, Balaji Krishnamurthy* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524) -->


<!-- 1. **Correlated topic models** *John Lafferty, David Blei* `NeurIPS 2005` [[pdf]]([https://arxiv.org/abs/2012.01524](https://proceedings.neurips.cc/paper_files/paper/2005/hash/9e82757e9a1c12cb710ad680db11f6f1-Abstract.html)) -->

<!-- 1. **Probabilistic topic models** *David Blei* `Communications of the ACM 2012` [[pdf]](https://arxiv.org/abs/2012.01524) -->

<!-- 1. **Dynamic topic models** *David Blei, John Lafferty* `ICML 2006` [[pdf]](https://arxiv.org/abs/2012.01524) -->

<!-- 1. **Latent dirichlet allocation** *David Blei, Andrew Ng, Michael Jordan* `JLMR 2003` [[pdf]](https://arxiv.org/abs/2012.01524) -->

<!-- 1. **The nested chinese restaurant process and bayesian nonparametric inference of topic hierarchies.** *David Blei, Griffiths TL, Michael Jordan* `JACM 2010` [[pdf]](https://arxiv.org/abs/2012.01524) -->

<!-- 1. **Hierarchical topic models and the nested chinese restaurant process** *David M. Blei, Michael I. Jordan, Thomas L. Griffiths, Joshua B. Tenenbaum* `NeurIPS 2023` [[pdf]](https://arxiv.org/abs/2012.01524) -->


<!-- 1. **Kernel topic models** *Philipp Hennig, David Stern, Ralf Herbrich, Thore Graepel* `arXiv 2011` [[pdf]](https://arxiv.org/abs/1110.4713) -->

<!-- 1. **Replicated softmax: an undirected topic model** *Geoffrey E. Hinton, Russ R. Salakhutdinov* `NeurIPS 2009` [[pdf]](https://papers.nips.cc/paper_files/paper/2009/hash/31839b036f63806cba3f47b93af8ccb5-Abstract.html) -->

<!-- 1. **Simultaneous discovery of common and discriminative topics via joint nonnegative matrix factorization** *Hannah Kim, Jaegul Choo, Jingu Kim, Chandan K. Reddy, Haesun Park* `KDD 2015` [[pdf]](https://dl.acm.org/doi/10.1145/2783258.2783338) -->


<!-- 1. **A systematic review of the use of topic models for short text social media analysis** *Caitlin Doogan Poet Laureate, Wray Buntine, Henry Linger* `Artificial Intelligence Review 2023` [[pdf]](https://link.springer.com/article/10.1007/s10462-023-10471-x) -->


<!-- 1. **Plda+ parallel latent dirichlet allocation with data placement and pipeline processing** *Zhiyuan Liu, Yuzhou Zhang, Edward Y. Chang, Maosong Sun* `ACM Transactions on Intelligent Systems and Technology 2011` [[pdf]](https://dl.acm.org/doi/10.1145/1961189.1961198) -->

<!-- 1. **Supervised topic models** *David M. Blei, Jon D. McAuliffe* `NeurIPS 2007` [[pdf]](https://papers.nips.cc/paper_files/paper/2007/hash/d56b9fc4b0f1be8871f5e1c40c0067e7-Abstract.html) -->


<!-- 1. **Polylingual topic models** *David Mimno, Hanna M. Wallach, Jason Naradowsky, David A. Smith, Andrew McCallum* `EMNLP 2009` [[pdf]](https://aclanthology.org/D09-1092/) -->

<!-- 1. **Optimizing semantic coherence in topic models** *David Mimno, Hanna Wallach, Edmund Talley, Miriam Leenders, Andrew McCallum* `EMNLP 2011` [[pdf]](https://dl.acm.org/doi/10.5555/2145432.2145462) -->


<!-- 1. **Distributed algorithms for topic models** *David Newman, Arthur Asuncion, Padhraic Smyth, Max Welling* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2104.05064) -->

<!-- 1. **Neural discrete representation learning** *Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu* `arXiv 2017` [[pdf]](https://arxiv.org/abs/1711.00937) -->

<!-- 1. **Representation learning with contrastive predictive coding** *Aaron van den Oord, Yazhe Li, Oriol Vinyals* `arXiv 2018` [[pdf]](https://arxiv.org/abs/1807.03748) -->

<!-- 1. **Collaborative topic modeling for recommending scientific articles** *Chong Wang, David M. Blei* `KDD 2011` [[pdf]](https://dl.acm.org/doi/10.1145/2020408.2020480) -->


<!-- 1. **Plda: Parallel latent dirichlet allocation for large-scale applications** *Yi Wang, Hongjie Bai, Matt Stanton, Wen-Yen Chen, Edward Y. Chang* `AAIM 2009` [[pdf]](https://link.springer.com/chapter/10.1007/978-3-642-02158-9_26) -->


<!-- 1. **A biterm topic model for short texts** *Xiaohui Yan, Jiafeng Guo, Yanyan Lan, Xueqi Cheng* `WWW 2013` [[pdf]](https://dl.acm.org/doi/10.1145/2488388.2488514) -->

<!-- 1. **A dirichlet multinomial mixture model-based approach for short text clustering** *Jianhua Yin, Jianyong Wang* `KDD 2014` [[pdf]](https://dl.acm.org/doi/10.1145/2623330.2623715) -->

<!-- 1. **Multilingual anchoring: Interactive topic modeling and alignment across languages** *Michelle Yuan, Benjamin Van Durme, Jordan L. Ying* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524) -->




## NTMs with Different Structures

### VAE-based NTMs

1. **Neural Variational Inference for Text Processing** *Yishu Miao, Lei Yu, Phil Blunsom* `ICML 2016` [[pdf]](https://arxiv.org/pdf/1511.06038) [[code]](https://github.com/ysmiao/nvdm)

1. **Autoencoding Variational Inference For Topic Models** *Akash Srivastava, Charles Sutton* `ICLR 2017` [[pdf]](https://arxiv.org/pdf/1703.01488) [[code]](https://github.com/akashgit/autoencoding_vi_for_topic_models)

1. **Discovering Discrete Latent Topics with Neural Variational Inference** *Yishu Miao, Edward Grefenstette, Phil Blunsom* `ICML 2017` [[pdf]](http://proceedings.mlr.press/v70/miao17a/miao17a.pdf)

1. **Neural Models for Documents with Metadata** *Dallas Card, Chenhao Tan, Noah A. Smith* `ACL 2018` [[pdf]](https://arxiv.org/pdf/1705.09296) [[code]](https://github.com/dallascard/scholar)

1. **Coherence-aware Neural Topic Modeling** *Ran Ding, Ramesh Nallapati, Bing Xiang* `EMNLP 2018` [[pdf]](https://arxiv.org/abs/1809.02687) [[code]](https://paperswithcode.com/paper/coherence-aware-neural-topic-modeling)

1. **A Discrete Variational Recurrent Topic Model without the Reparametrization Trick** *Mehdi Rezaee, Francis Ferraro* `NeurIPS 2020` [[pdf]](https://proceedings.neurips.cc/paper/2020/file/9f1d5659d5880fb427f6e04ae500fc25-Paper.pdf)

1. **Topic Modeling using Variational Auto-Encoders with Gumbel-Softmax and Logistic-Normal Mixture Distributions** *Denys Silveira, Andr’e Carvalho, MarcoCristo, Marie-FrancineMoens* `IJCNN 2018` [[pdf]](https://ieeexplore.ieee.org/abstract/document/8489778)

1. **Improving Topic Quality by Promoting Named Entities in Topic Modeling** *Katsiaryna Krasnashchok, Salim Jouili* `ACL 2018` [[pdf]](https://aclanthology.org/P18-2040/)

1. **TAN-NTM: Topic Attention Networks for Neural Topic Modeling** *Madhur Panwar, Shashank Shailabh, Milan Aggarwal, Balaji Krishnamurthy* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)


### NTMs with Various Priors

1. **Discovering Discrete Latent Topics with Neural Variational Inference** *Yishu Miao, Edward Grefenstette, Phil Blunsom* `ICML 2017` [[pdf]](http://proceedings.mlr.press/v70/miao17a/miao17a.pdf)

1. **Dirichlet variational autoencoder** *Weonyoung Joo, Wonsung Lee, Sungrae Park, Il-Chul Moon* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1901.02739)

1. **Decoupling Sparsity and Smoothness in the Dirichlet Variational Autoencoder Topic Model** *Sophie Burkhardt, Stefan Kramer* `JMLR 2019` [[pdf]](https://www.jmlr.org/papers/volume20/18-569/18-569.pdf)

1. **WHAI: Weibull Hybrid Autoencoding Inference for Deep Topic Modeling** *Hao Zhang, Bo Chen, Dandan Guo, Mingyuan Zhou* `ICLR 2018` [[pdf]](https://arxiv.org/pdf/1803.01328) [[code]](https://github.com/BoChenGroup/WHAI)

1. **Learning VAE-LDA models with Rounded Reparameterization Trick** *Runzhi Tian, Yongyi Mao, Richong Zhang* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.101/)




### NTMs with Embeddings

1. **Topic Modeling in Embedding Spaces** *Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei* `TACL 2020` [[pdf]](https://aclanthology.org/2020.tacl-1.29.pdf) [[code]](https://paperswithcode.com/paper/?acl=2020.tacl-1.29)

1. **Neural topic model via optimal transport** *He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine* `ICLR 2021` [[pdf]](https://arxiv.org/abs/2008.13537)

1. **Representing mixtures of word embeddings with mixtures of topic embeddings** *Dongsheng Wang, Dandan Guo, He Zhao, Huangjie Zheng, Korawat Tanwisuth, Bo Chen, Mingyuan Zhou* `ICLR 2022` [[pdf]](https://arxiv.org/abs/2203.01570)

1. **Hyperminer: Topic taxonomy mining with hyperbolic embedding** *Yishi Xu, Dongsheng Wang, Bo Chen, Ruiying Lu, Zhibin Duan, Mingyuan Zhou* `arXiv 2022` [[pdf]](https://arxiv.org/abs/2210.10625)

1. **Effective Neural Topic Modeling with Embedding Clustering regularization** *Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu* `ICML 2023` 



### NTMs with PLMs

1. **Improving Neural Topic Models using Knowledge Distillation** *Alexander Miserlis Hoyle, Pranav Goel, Philip Resnik* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.137.pdf) [[code]](https://paperswithcode.com/paper/?acl=2020.emnlp-main.137)


1. **Cross-lingual Contextualized Topic Models with Zero-shot Learning** *Federico Bianchi, Silvia Terragni, Dirk Hovy, Debora Nozza, Elisabetta Fersini* `EACL 2021` [[pdf]](https://aclanthology.org/2021.eacl-main.143.pdf) [[code]](https://paperswithcode.com/paper/?acl=2021.eacl-main.143)


1. **Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning** *Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, William Yang Wang* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2301.11916)


### NTMs with Reinforcement Learning

1. **Neural topic model with reinforcement learning** *Lin Gui, Jia Leng, Gabriele Pergola, Yu Zhou, Ruifeng Xu, Yulan He* `EMNLP 2019` [[pdf]](https://aclanthology.org/D19-1350/)

1. **Reinforcement learning for topic models** *Jeremy Costello, Marek Z. Reformat* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2012.01524)


### NTMs with Adversarial Training

1. **ATM: Adversarial-neural Topic Model** *Rui Wang, Deyu Zhou, Yulan He* `Information Processing & Management 2019` [[pdf]](https://arxiv.org/pdf/1811.00265)

1. **Neural Topic Modeling with Bidirectional Adversarial Training** *Rui Wang, Xuemeng Hu, Deyu Zhou, Yulan He, Yuxuan Xiong, Chenchen Ye, Haiyang Xu* `ACL 2020` [[pdf]](https://aclanthology.org/2020.acl-main.32/) [[code]](https://github.com/zll17/Neural_Topic_Models)

1. **Neural Topic Modeling with Cycle-consistent Adversarial Training** *Xuemeng Hu, Rui Wang, Deyu Zhou, Yuxuan Xiong* `EMNLP 2020` [[pdf]](https://arxiv.org/pdf/2009.13971)


### NTMs with Contrastive Learning

1. **Improving topic disentanglement via contrastive learning** *Xixi Zhou, Jiajin Bu, Sheng Zhou, Zhi Yu, Ji Zhao, Xifeng Yan* `ACM Information Processing and Management: an International Journal` [[pdf]](https://dl.acm.org/doi/10.1016/j.ipm.2022.103164)

1. **Contrastive Learning for Neural Topic Model** *Thong Nguyen, Anh Tuan Luu* `NeurIPS 2021` [[pdf]](https://arxiv.org/pdf/2110.12764) [[code]](https://github.com/nguyentthong/CLNTM)


### NTMs with Metadata

1. **Discriminative topic modeling with logistic lda** *Iryna Korshunova, Hanchen Xiong, Mateusz Fedoryszak, Lucas Theis* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1909.01436)

1. **Layer-assisted neural topic modeling over document networks** *Yiming Wang, Ximing Li, Jihong Ouyang* `IJCAI 2021` [[pdf]](https://www.ijcai.org/proceedings/2021/0433)

1. **Neural topic model with attention for supervised learning** *Xinyi Wang, Yi Yang* `AISTATS 2020` [[pdf]](https://proceedings.mlr.press/v108/wang20c.html)


### NTMs with Graphs

1. **GraphBTM: Graph Enhanced Autoencoded Variational Inference for Biterm Topic Model** *Qile Zhu, Zheng Feng, Xiaolin Li* `EMNLP 2018` [[pdf]](https://aclanthology.org/D18-1495.pdf) [[code]](https://github.com/valdersoul/GraphBTM)

1. **Variational Graph Author Topic Modeling** *Delvin Ce Zhang, Hady W. Lauw* `KDD 2022` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539310?casa_token=mPNGXYJm5hwAAAAA:8J4aSzN7dXmFaT98f13LVh4oF4p1mKm4UZJ_jAQPgcgyfXDOs9YEGpR6Zz_X-eK6LOWcbRCJ0Vdjf2M)

1. **Topic Modeling on Document Networks with Adjacent-Encoder** *Delvin Ce Zhang, Hady W. Lauw* `AAAI 2022` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/6152/6008) [[code]](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcezhang01%2FAdjacent-Encoder&sa=D&sntz=1&usg=AOvVaw3jGZDHrfjxic8x3teUK0fh)

1. **Neural Topic Modeling by Incorporating Document Relationship Graph** *Deyu Zhou, Xuemeng Hu, Rui Wang* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.310/) 

1. **Graph Attention Topic Modeling Network** *Liang Yang, Fan Wu, Junhua Gu, Chuan Wang, Xiaochun Cao, Di Jin, Yuanfang Guo* `WWW 2020` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3366423.3380102?casa_token=8fqM-QD88WUAAAAA:E3uNrpuXWVC_4Kd1nZ-fpSrd3_mxClzEx_FY23lsqaHLDryXdsK3NINRPSk4BATi7jJZqSJHP5ewjg)

1. **Graph neural topic model with commonsense knowledge** *Bingshan Zhu, Yi Cai, Haopeng Ren* `Information Processing & Management 2023` [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0306457322003168)



### Other NTMs

1. **A Neural Autoregressive Topic Model** *Hugo Larochelle, Stanislas Lauly* `NeurIPS 2012` [[pdf]](https://papers.nips.cc/paper/2012/hash/b495ce63ede0f4efc9eec62cb947c162-Abstract.html) 

1. **A Novel Neural Topic Model and its Supervised Extension** * Ziqiang Cao, Sujian Li, Yang Liu, Wenjie Li, Heng Ji* `AAAI 2015` [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/9499/9358)

1. **Document informed neural autoregressive topic models with distributional prior** *Pankaj Gupta, Yatin Chaudhary, Florian Buettner, Hinrich Schütze* `AAAI 2019` [[pdf]](https://arxiv.org/abs/1809.06709)

1. **TextTOvec: Deep Contextualized Neural Autoregressive Topic Models of Language with Distributed Compositional Prior** * Pankaj Gupta, Yatin Chaudhary, Florian Buettner, Hinrich Schütze* ` ICLR 2019` [[pdf]](https://arxiv.org/abs/1810.03947)

1. **Sparsemax and relaxed wasserstein for topic sparsity** *Tianyi Lin, Zhiyue Hu, Xin Guo* `WSDM 2019` [[pdf]](https://arxiv.org/abs/1810.09079)

1. **Topic Modeling with Wasserstein Autoencoders** *Feng Nan, Ran Ding, Ramesh Nallapati, Bing Xiang* `ACL 2019` [[pdf]](https://aclanthology.org/P19-1640.pdf) [[code]](https://paperswithcode.com/paper/?acl=P19-1640)

1. **Discovering Topics in Long-tailed Corpora with Causal Intervention** *Xiaobao Wu, Chunping Li, Yishu Miao* `ACL 2021 findings` [[pdf]](https://aclanthology.org/2021.findings-acl.15.pdf) [[code]](https://github.com/bobxwu/DecTM)




## NTMs for Various Scenarios

### Correlated NTMs

1. **Neural Variational Correlated Topic Modeling** *Luyang Liu, Heyan Huang, Yang Gao, Yongfeng Zhang, Xiaochi Wei* `WWW 2019` [[pdf]](https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313561)


### Hierarchical NTMs

1. **Neural attention-aware hierarchical topic model** *Yuan Jin, He Zhao, Ming Liu, Lan Du, Wray Buntine* `arXiv 2021` [[pdf]](https://arxiv.org/abs/2110.07161)

1. **Neural topic models for hierarchical topic detection and visualization** *Dang Pham, Tuan M. V. Le* `ECML PKDD 2021` [[pdf]](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_219.pdf)


### Short Text NTMs

1. **Copula Guided Neural Topic Modelling for Short Texts** *Lihui Lin, Hongyu Jiang, Yanghui Rao* `SIGIR 2020` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401245) [[code]](https://github.com/linkstrife/CR-GSM-NVCTM)

1. **Context reinforced neural topic modeling over short texts** *Jiachun Feng, Zusheng Zhang, Cheng Ding, Yanghui Rao, Haoran Xie* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2008.04545)

1. **Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder** *Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.138.pdf) [[code]](https://github.com/bobxwu/NQTM)

1. **Extracting Topics with Simultaneous Word Co-occurrence and Semantic Correlation Graphs: Neural Topic Modeling for Short Texts** *Yiming Wang, Ximing Li, Xiaotang Zhou, Jihong Ouyang* `EMNLP 2021 findings` [[pdf]](https://aclanthology.org/2021.findings-emnlp.2.pdf)

1. **A neural topic model with word vectors and entity vectors for short texts** *Xiaowei Zhao, Deqing Wang, Zhengyang Zhao, Wei Liu, Chenwei Lu, Fuzhen Zhuang* `Information Processing & Management 2021` [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S030645732030947X)

1. **Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning** *Xiaobao Wu, Anh Tuan Luu, Xinshuai Dong* `EMNLP 2022` [[pdf]]() [[code]](https://github.com/bobxwu/TSCTM)



### Cross-lingual NTMs

1. **Learning Multilingual Topics with Neural Variational Inference** *Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao* `NLPCC 2020` [[pdf]](https://bobxwu.github.io/files/pub/NLPCC2020_Neural_Multilingual_Topic_Model.pdf) [[code]](https://github.com/BobXWu/NMTM)

1. **Multilingual and Multimodal Topic Modelling with Pretrained Embeddings** *Elaine Zosa, Lidia Pivovarova* `COLING 2022`  [[pdf]](https://researchportal.helsinki.fi/files/228080474/COLING_2022_M3L_Topic_Modelling.pdf) [[code]](https://github.com/ezosa/M3L-topic-model)

1. **Infoctm: A mutual information maximization perspective of cross-lingual topic modeling** *Xiaobao Wu, Xinshuai Dong, Thong Nguyen, Chaoqun Liu, Liangming Pan, Anh Tuan Luu* `AAAI 2023` [[pdf]](https://arxiv.org/abs/2304.03544)

1. **Fine-tuning encoders for improved monolingual and zeroshot polylingual neural topic modeling** *Aaron Mueller, Mark Dredze* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)



### Hierarchical NTMs

1. **Tree-Structured Topic Modeling with Nonparametric Neural Variational Inference** *Ziye Chen, Cheng Ding, Zusheng Zhang, Yanghui Rao, Haoran Xie* `ACL 2021` [[pdf]](https://aclanthology.org/2021.acl-long.182/)

<!-- 1. **Tree-Structured Decoding with DOUBLYRECURRENT Neural Networks** *David Alvarez-Melis, Tommi S. Jaakkola* `ICLR 2017` [[pdf]](https://openreview.net/forum?id=HkYhZDqxg) -->

1. **Tree-structured Neural Topic Model** *Masaru Isonuma, Junichiro Mori, Danushka Bollegala, Ichiro Sakata* `ACL 2020` [[pdf]](https://aclanthology.org/2020.acl-main.73.pdf)


### Dynamic NTMs

1. **The dynamic embedded topic model** *Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei* `arXiv 2019` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Dynamic Topic Models for Temporal Document Networks** *Delvin Ce Zhang, Hady W. Lauw* `ICML 2022` [[pdf]](https://proceedings.mlr.press/v162/zhang22n/zhang22n.pdf)

1. **Neural dynamic focused topic model** *Kostadin Cvejoski, Ramsés J. Sánchez, César Ojeda* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Antm: An aligned neural topic model for exploring evolving topics** *Hamed Rahimi, Hubert Naacke, Camelia Constantin, Bernd Amann* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2302.01501)



### Lifelong NTMs

1. **Neural topic modeling with continual lifelong learning** *Pankaj Gupta, Yatin Chaudhary, Thomas Runkler, Hinrich Schütze* `ICML 2020` [[pdf]](https://arxiv.org/abs/2006.10909)

1. **Lifelong topic modeling with knowledge-enhanced adversarial network** *Xuewen Zhang, Yanghui Rao, Qing Li* `WWW 2022` [[pdf]](https://link.springer.com/article/10.1007/s11280-021-00984-2)



## Discovering Topics by Clustering

Note that some studies are not real topic models since they can only produce topics while cannot infer the topic distributions of documents as required.


1. **Topic Modeling with Contextualized Word Representation Clusters** *Laure Thompson and David Mimno* `arXiv 2020` [[pdf]](https://arxiv.org/pdf/2010.12626)

1. **Top2vec: Distributed representations of topics** *Dimo Angelov* `arXiv 2020` [[pdf]]([https://arxiv.org/abs/2012.01524](https://arxiv.org/abs/2008.09470))

1. **Topic modeling with contextualized word representation clusters** *Laure Thompson, David Mimno* `ACL 2020` [[pdf]](https://arxiv.org/abs/2010.12626)

1. **Tired of topic models? clusters of pretrained word embeddings make for fast and good topics too** *Suzanna Sia, Ayush Dalmia, Sabrina J. Mielke* `EMNLP 2020` [[pdf]](https://aclanthology.org/2020.emnlp-main.135/)

1. **Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence** *Federico Bianchi, Silvia Terragni, Dirk Hovy* `ACL 2021` [[pdf]](https://aclanthology.org/2021.acl-short.96.pdf) [[code]](https://paperswithcode.com/paper/?acl=2021.acl-short.96)

1. **BERTopic: Neural topic modeling with a class-based TF-IDF procedure** *Maarten Grootendorst* `arXiv 2022` [[pdf]](https://arxiv.org/abs/2203.05794)

1. **Is neural topic modelling better than clustering? an empirical study on clustering with contextual embeddings for topics** *Zihan Zhang, Meng Fang, Ling Chen, Mohammad-Reza Namazi-Rad* `NAACL 2022` [[pdf]](https://aclanthology.org/2022.naacl-main.285/)

1. **Effective Seed-Guided Topic Discovery by Integrating Multiple Types of Contexts** *Yu Zhang, Yunyi Zhang, Martin Michalski, Yucheng Jiang, Yu Meng, Jiawei Han* `WSDM 2023` [[pdf]](https://arxiv.org/abs/2212.06002)



## Applications of NTMs

### Text Analysis

1. **Topic Memory Networks for Short Text Classification** *Jichuan Zeng, Jing Li, Yan Song, Cuiyun Gao, Michael R. Lyu, Irwin King* `EMNLP 2018` [[pdf]](https://aclanthology.org/D18-1351.pdf) [[code]](https://github.com/zengjichuan/TMN)

1. **Neural Relational Topic Models for Scientific Article Analysis** *Haoli Bai, Zhuangbin Chen, Michael R. Lyu, Irwin King, Zenglin Xu* `CIKM 2018` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3269206.3271696?casa_token=Ak1DPOATCdQAAAAA:5i7CAa9x2dA8XUoN6ZBBLvKinqfR9OsqskT5ZlyJVNQ2vJvfv73q7eIeMqEvxrViP36PtohbB70wtg)

1. **Interaction-aware Topic Model for Microblog Conversations through Network Embedding and User Attention** *Ruifang He, Xuefei Zhang, Di Jin, Longbiao Wang, Jianwu Dang, Xiangang Li* `COLING 2018` [[pdf]](https://aclanthology.org/C18-1118/) 

1. **TopicBERT for Energy Efficient Document Classification** *Yatin Chaudhary, Pankaj Gupta, Khushbu Saxena, Vivek Kulkarni, Thomas Runkler, Hinrich Schütze* `EMNLP 2020 Findings` [[pdf]](https://arxiv.org/abs/2010.16407) [[code]](https://github.com/YatinChaudhary/TopicBERT)

1. **Topic modeling techniques for text mining over a large-scale scientific and biomedical text corpus** *Sandhya Avasthi, Ritu Chauhan, Debi Prasanna Acharjya* `IJACI 2022` [[pdf]]([https://arxiv.org/abs/2012.01524](https://www.igi-global.com/article/topic-modeling-techniques-for-text-mining-over-a-large-scale-scientific-and-biomedical-text-corpus/293137))

1. **Classification aware neural topic model for covid-19 disinformation categorisation** *Xingyi Song, Johann Petrak, Ye Jiang, Iknoor Singh, Diana Maynard, Kalina Bontcheva* `PLOS 2021` [[pdf]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0247086)

1. **Topic modeling on podcast short-text metadata** *Francisco B. Valero, Marion Baranes, Elena V. Epure* `ECIR 2022` [[pdf]](https://arxiv.org/abs/2201.04419)

1. **Multi task mutual learning for joint sentiment classification and topic detection** *Lin Gui; Jia Leng; Jiyun Zhou; Ruifeng Xu; Yulan He* `IEEE Transactions on Knowledge and Data Engineering 2020` [[pdf]](https://ieeexplore.ieee.org/document/9112648)


1. **Topic Modeling for Multi-Aspect Listwise Comparisons** *Delvin Ce Zhang, Hady W. Lauw* `CIKM 2021` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3459637.3482398?casa_token=iLrtUseO1P8AAAAA:RgA_5uYNsLR2WSxVYa_6VSSI6ZxJitmBlgcmVSLGznWa_auqE3IHP8S5zO-nWM6L5r3OKQ81N8Ss6Xg) [[code]](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcezhang01%2Fmalic&sa=D&sntz=1&usg=AOvVaw2Bqy_hLzRu13FRQdad_y_b)


### Text Generation

1. **A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features** *Hongyin Tang, Miao Li, Beihong Jin* `EMNLP 2019` [[pdf]](https://aclanthology.org/D19-1513/)

1. **Topic-Guided Abstractive Text Summarization: a Joint Learning Approach** *Chujie Zheng, Kunpeng Zhang, Harry Jiannan Wang, Ling Fan, Zhe Wang* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2010.10323)

1. **What you say and how you say it: Joint modeling of topics and discourse in microblog conversations** *Jichuan Zeng, Jing Li, Yulan He, Cuiyun Gao, Michael R. Lyu, Irwin King* `TACL 2019` [[pdf]](https://arxiv.org/abs/1903.07319)

1. **Enriching and Controlling Global Semantics for Text Summarization** *Thong Nguyen, Anh Tuan Luu, Truc Lu, Tho Quan* `EMNLP 2021` [[pdf]](https://aclanthology.org/2021.emnlp-main.744/)

1. **Enhancing extractive text summarization with topic-aware graph neural networks** *Peng Cui, Le Hu, Yuanchao Liu* `COLING 2020` [[pdf]](https://arxiv.org/abs/2010.06253)

1. **Recurrent Hierarchical Topic-guided RNN for Language Generation** *Dandan Guo, Bo Chen, Ruiying Lu, Mingyuan Zhou* `ICML 2020` [[pdf]](https://arxiv.org/abs/1912.10337)

1. **Topnet: Learning from neural topic model to generate long stories** *Yazheng Yang, Boyuan Pan, Deng Cai, Huan Sun* `KDD 2021` [[pdf]](https://arxiv.org/abs/2112.07259)

1. **HTKG: Deep Keyphrase Generation with Neural Hierarchical Topic Guidance** *Yuxiang Zhang, Tao Jiang, Tianyu Yang, Xiaoli Li, Suge Wang* `SIGIR 2022` [[pdf]](https://dl.acm.org/doi/abs/10.1145/3477495.3531990)

1. **TopicRNN: A Recurrent Neural Network with Long-range Semantic Dependency** *Adji B. Dieng, Chong Wang, Jianfeng Gao, John Paisley* `ICLR 2017` [[pdf]](https://arxiv.org/abs/1611.01702) [[code]](https://paperswithcode.com/paper/topicrnn-a-recurrent-neural-network-with-long)


### Content Recommendation

1. **Structured Neural Topic Models for Reviews** *Babak Esmaeili, Hongyi Huang, Byron C. Wallace, Jan-Willem van de Meent* `AISTATS 2019` [[pdf]](https://arxiv.org/abs/1812.05035) 

1. **Graph neural collaborative topic model for citation recommendation** *Qianqian Xie, Yutao Zhu, Jimin Huang, Pan Du, Jian-Yun Nie* `ACM Transactions on Information Systems 2021` [[pdf]](https://dl.acm.org/doi/10.1145/3473973?sid=SCITRUS)




## Evaluation of Topic Models

1. **Evaluation Methods for Topic Models** *Hanna M. Wallach, Iain Murray, Ruslan Salakhutdinov, David Mimno* `ICML 2009` [[pdf]](http://dirichlet.net/pdf/wallach09evaluation.pdf)

1. **Reading tea leaves: How humans interpret topic models** *Jonathan Chang, Sean Gerrish, Chong Wang, Jordan Boyd-graber, David Blei* `NeurIPS 2009` [[pdf]](https://papers.nips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html)

1. **Estimating likelihoods for topic models** *Wray Buntine* `ACML 2009` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Automatic evaluation of topic coherence** *David Newman, Jey Han Lau, Karl Grieser, Timothy Baldwin* `NAACL 2010` [[pdf]](https://aclanthology.org/N10-1012/)

1. **Topic Model or Topic Twaddle? Re-evaluating Semantic Interpretability Measures** *Caitlin Doogan, Wray Buntine* `NAACL 2021` [[pdf]](https://aclanthology.org/2021.naacl-main.300/)

1. **Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality** *Jey Han Lau, David Newman, Timothy Baldwin* `ACL 2014` [[pdf]](https://aclanthology.org/E14-1056/) [[code]](https://github.com/jhlau/topic_interpretability)

1. **Exploring the Space of Topic Coherence Measures** *Michael Röder, Andreas Both, Alexander Hinneburg* `WSDM 2015` [[pdf]](https://dl.acm.org/doi/pdf/10.1145/2684822.2685324?casa_token=SZCz7HIe8ecAAAAA:w76e2OqcMLJ6lcuTkU050S_QREP8LNm2kAXpV-O47kAT6FW9jpsBwMp-2Vsa_iDxVxpV0LfkoQSZGA) [[code]](https://github.com/dice-group/Palmetto)

1. **Is automated topic model evaluation broken? the incoherence of coherence** *Alexander Hoyle, Pranav Goel, Denis Peskov, Andrew Hian-Cheong, Jordan Boyd-Graber, Philip Resnik* `NeurIPS 2021` [[pdf]](https://arxiv.org/abs/2107.02173)

1. **Are Neural Topic Models Broken?** *Alexander Hoyle, Pranav Goel, Rupak Sarkar, Philip Resnik* `EMNLP 2022` [[pdf]](https://arxiv.org/abs/2210.16162)

1. **Benchmarking neural topic models: An empirical study** *Thanh-Nam Doan, Tuan-Anh Hoang* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.01524)

1. **Re-visiting automated topic model evaluation with large language models** *Dominik Stammbach, Vilém Zouhar, Alexander Hoyle, Mrinmaya Sachan, Elliott Ash* `arXiv 2023` [[pdf]](https://arxiv.org/abs/2305.12152)

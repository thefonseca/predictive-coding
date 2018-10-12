# Deep Predictive Coding for Multimodal Representation Learning

## Abstract

In machine learning parlance, common sense reasoning relates to the capacity of _learning representations_ that disentangle hidden factors behind spatiotemporal sensory data. In this work, we hypothesise that the predictive coding theory of perception and learning from neuroscience literature may be a good candidate for implementing such common sense inductive biases. We build upon a previous deep learning implementation of predictive coding by [Lotter et al., 2016](#lotter) and extend its application to the challenging task of inferring abstract, everyday human actions such as _cooking_ and _diving_. Furthermore, we propose a novel application of the same architecture to process auditory data, and find that with a simple sensory substitution trick, the predictive coding model can learning useful representations. Our transfer learning experiments also demonstrate good generalisation of learned representations on the UCF-101 action classification dataset.

## Relevant documents
* [Project proposal](./informatics-project-proposal.pdf)
* [Project progress report](./project-progress-report.pdf)
* [Latest dissertation version](https://github.com/thefonseca/msc-project/raw/master/dissertation.pdf)

## Project folders
* [datasets](./datasets): includes scripts for downloading and preprocessing of the datasets used in the experiments, including the Moments in Time and UCF-101 datasets.
* [models/prednet](./models/prednet): the primary model implementation for our study. The model code is adapted from the implementation provided by [Lotter, 2016](#lotter). All the pipeline was reimplemented to fit our experimental needs.
* [models/classifier](./models/classifier): implementation of simple SVM and LSTM classifiers used on top of predictive coding representations.

## References

##### Friston
Friston, K., & Kiebel, S. (2009). [Predictive coding under the free-energy principle](http://rstb.royalsocietypublishing.org/content/364/1521/1211). Philosophical Transactions of the Royal Society B: Biological Sciences, 364(1521), 1211-1221.

##### Lotter
Lotter, W., Kreiman, G., & Cox, D. (2016). [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104). arXiv preprint arXiv:1605.08104.

##### Frermann
Frermann, L., Cohen, S. B., & Lapata, M. (2017). [Whodunnit? Crime Drama as a Case for Natural Language Understanding](https://arxiv.org/abs/1710.11601). arXiv preprint arXiv:1710.11601.

##### Hermann
Hermann, K. M., Hill, F., Green, S., Wang, F., Faulkner, R., Soyer, H., ... & Wainwright, M. (2017). [Grounded language learning in a simulated 3D world](https://arxiv.org/abs/1706.06551). arXiv preprint arXiv:1706.06551.

##### Monfort
Monfort, M., Zhou, B., Bargal, S. A., Andonian, A., Yan, T., Ramakrishnan, K., ... & Oliva, A. (2018). [Moments in Time Dataset: one million videos for event understanding](https://arxiv.org/abs/1801.03150). arXiv preprint arXiv:1801.03150.

##### Aytar
Aytar, Y., Vondrick, C., & Torralba, A. (2017). [See, hear, and read: Deep aligned representations](https://arxiv.org/abs/1706.00932). arXiv preprint arXiv:1706.00932.

# Repertoire Analysis and Classification
We have developed the ERGO models in order to detect binding of specific TCRs and peptides.
The next level will be applying those methods in order to classify a full TCR repertoire.

## Report 9.7.19 :computer:
ERGO original code was modified for test predictions.
(We now use pytorch batching and loading methods, it is neater.)

## Report 11.7.19 :pencil2:
### High ERGO Scores on CMV Repertoires
ERGO trained model can predict a binding score for every given TCR and peptide.
In order to classify a repertoire, we will use these scores for every TCR in the repertoire.
We are using Emerson et al. CMV associated repertoires (CMV+ and CMV-), together with CMV peptides in McPAS-TCR database,
mainly NLVPMVATV (the most frequent one).

Given a repertoire, we pair every TCR with the CMV peptide. Next, we check for predicted ERGO scores of the pairs.
The tables below show how many TCRs in a repertoire get a high CMV-binding score:

pathology | model | repertoire size | number of TCRs with score > 0.99 | score > 0.999
--- | --- | --- | --- | ---
CMV+| LSTM | 159833 | 4305 (2.69%) | 745 (0.466%)
CMV-| LSTM | 104186 | 2835 (2.72%) | 500 (0.479%)
CMV+| AE   | 159833 |    | 2341 (1.46%)
CMV-| AE   | 104186 |    | 1686 (1.61%)

Currently it is hard to see any difference between repertoires.

## Report 16.7.19 :bar_chart:
One repertoire might not be enough, so we took 10 CMV+ and 10 CMV- repertoires.
![10 repertoires scores > 0.999](https://github.com/IdoSpringer/repertoire_classification/blob/master/plots/high_scores_10_repertoires.png)

The CMV+ and CMV- distributions still look very similar.
We looked over the whole histograms:
![](https://github.com/IdoSpringer/repertoire_classification/blob/master/plots/hists_high_scores_cmv.png)

The model learns to dump the scores to the edges (thanks to cross-entropy loss).
We are interested in the highest bin of the histogram. We expect that CMV+ repertoires will be
closer to the '1' edge.
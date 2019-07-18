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
![10 repertoires scores > 0.999](plots/high_scores_10_repertoires.png)

The CMV+ and CMV- distributions still look very similar.
We looked over the whole histograms:
![](plots/hists_high_scores_cmv.png)

The model learns to dump the scores to the edges (thanks to cross-entropy loss).
We are interested in the highest bin of the histogram. We expect that CMV+ repertoires will be
closer to the '1' edge.

(The histograms above are not normalized and are affected by the repertoire size)

Since only the last bin is important, we looked for a way to examine it.
We took +-log(1-bin) histograms for the highest bin (score > 0.98) in every repertoire
(because it should be close to 1).
![](plots/highest_bin_norm_hists.png)

Now the histograms are normalized.
The left side is CMV- repertoires, and the right side is CMV+.
Again, currently we do not see major differences.

## Report 18.7.19 :scroll:

### Saving ERGO predictions
We saved ERGO model predictions for the current repertoires (10 CMV+ and 10 CMV-),
paired with several CMV peptides.
Next week we will try to extract the predictions for all repertoires in Emerson et al. data, and for other
pathology-associated databases.

### Plotted histograms
We would like to compare between the CMV+ and CMV- histograms showed earlier.
Given the log(1-x) of the highest bin histograms, 
the histograms were normalized to a density function for each repertoire (forgetting the repertoire size),
and then were plotted together, distinguishing CMV+ and CMV- repertoires.
![](plots/plot_hist_high_bin.png)
 

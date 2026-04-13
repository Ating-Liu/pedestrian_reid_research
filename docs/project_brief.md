# Project Brief

## Suggested Title

`Research on Global-Local Feature Fusion for Pedestrian Re-Identification`

## Resume Bullets

- Built a pedestrian re-identification training framework on `Market-1501`, `CUHK03-NP`, and `MSMT17`, covering dataset parsing, identity-balanced sampling, model training, evaluation, and retrieval visualization.
- Reproduced a strong `ResNet50 + BNNeck + CrossEntropy + Triplet Loss` baseline and extended it with a global-local dual-branch design for discriminative feature learning.
- Added a lightweight Transformer on local region tokens and an adaptive fusion gate to improve robustness under occlusion, viewpoint change, and background clutter.
- Designed a complete ablation workflow for `baseline / local branch / transformer branch / full model` and reported `Rank-1`, `Rank-5`, `Rank-10`, and `mAP`.

## Interview Notes

### Why Triplet Loss

Cross-entropy only separates training identities as a closed-set classification task. Re-identification is retrieval on unseen identities, so the embedding space also needs metric structure. Triplet loss reduces intra-class distance and enlarges inter-class distance, which makes ranking-based retrieval more stable.

### Why BNNeck

BNNeck decouples the optimization targets used by classification loss and metric loss. In practice, the classifier uses the batch-normalized feature while the metric loss uses the pre-BN embedding, which usually stabilizes training and improves retrieval performance.

### What Rank-1 and mAP Mean

- `Rank-1`: whether the top retrieved gallery image matches the query identity.
- `mAP`: averages precision over all correct matches in the ranked list, so it measures overall retrieval quality rather than only the first hit.

### Why the Local Branch Helps

Pedestrian re-identification is sensitive to occlusion, pose change, and background clutter. A purely global descriptor can miss stable cues such as shoes, bags, or upper-body texture. Local tokens preserve region-level details that complement the global descriptor.

### What the Transformer Module Solves Here

The Transformer is not used as a full backbone replacement. It is used only on local tokens to model long-range relations between body regions, which helps the network decide which local cues should be trusted together under partial occlusion or viewpoint change.

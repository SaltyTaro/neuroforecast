Assessment checked September 19, 2026. User constraint: public data only.

This document assesses the earlier chip-image proposal, which is now a fallback. The [subsequent topic hunt](topic_hunt.md) recommends DosePilot and includes its separate prior-work and novelty analysis. Do not transfer a novelty claim from one proposal to the other.

**The broad organ-on-a-chip quality-classification topic is not novel. Other participants could independently select it.** The public dataset and published examples make that a realistic competitive risk. Actual overlap with other entries is unknown: the competition API explicitly withholds public Writeups until the competition has fully closed.

The recommendation was driven by direct competition relevance, usable public data, and feasible implementation. Those strengths remain, but the proposed feature list does not yet establish a strong innovation claim.

| Proposed element | Verified prior work | Consequence for our claim |
| --- | --- | --- |
| Classifying chip quality from microscopy | [Supervised-Learning-Driven Interrogation of Organ-on-a-Chip Quality from Microscopy Images (2025)](https://doi.org/10.1021/cbe.5c00087), using a selected 631-image subset of the same source dataset | The central classification task is already published |
| Public images for automated chip evaluation | [Organ-On-A-Chip (OOC) Image Dataset for Machine Learning and Tissue Model Evaluation (2024)](https://doi.org/10.3390/data9020028) | The dataset's intended use already points entrants toward this application |
| Calibrated confidence | [On Calibration of Modern Neural Networks (2017)](https://proceedings.mlr.press/v70/guo17a.html) | Calibration is an established method, not our invention |
| Abstaining on uncertain inputs so a person can review them | [SelectiveNet: A Deep Neural Network with an Integrated Reject Option (2019)](https://proceedings.mlr.press/v97/geifman19a.html) | The underlying reject-option approach is established; this does not establish a specific human-review workflow in OoC |
| Synthetic-image augmentation as an alternative differentiator | [Synthetic Image Generation With a Fine-Tuned Latent Diffusion Model for Organ on Chip Cell Image Classification (2023)](https://doi.org/10.23919/SPA59660.2023.10274460) | This direction has also been explored. Title and publication metadata were verified through Crossref; full methods and results were not reviewed |

The 2025 quality-classification paper also explicitly proposes assessing multiple random splits and adding other cell types in future work. Broader validation and additional cell types alone would therefore be weak originality claims. Acquisition-group testing is a useful evaluation improvement, but the dataset lacks confirmed chip and donor identifiers; filename dates cannot establish independent-chip generalization. Our finding that all 59 date prefixes cross published splits warrants an audit, not an automatic claim of leakage or a novel method.

A potentially useful research question remains: **Can a review policy reduce expert-labeled bad images automatically accepted, at a fixed review workload, when evaluated on held-out acquisition dates or cell types?** This is a candidate empirical contribution. It is neither a measured result nor an established novelty claim. A serious evaluation would compare against a reproduced classifier, ordinary confidence thresholds, and established selective-prediction approaches under the same protocol. Improvements would need uncertainty estimates and clear limits on the biological conclusions supported by the labels.

An application of established methods can still be valuable in this judged competition. The current evidence supports keeping QC as a provisional feasibility candidate or fallback. It does not support committing to it as the strongest winning concept on originality grounds. The selection gate should require a specific gap in prior work, a meaningful result achievable with public data, and a credible connection to the competition's impact and innovation criteria (60% combined).

Search scope: the existing dataset paper and open 2025 article were checked, the 2023 conference metadata and the two primary ML-method pages were retrieved, and a focused Europe PMC title/abstract search was run. This establishes substantial prior overlap; it does not establish that any proposed narrower combination is new, or reveal other teams' hidden submissions.

# Out-of-distribution Data

This is a persistent and critical production issue present in any machine learning and deep learning systems, no matter how good the models were trained.

A single out-of-distribution sample data fed into a well-trained model can result in a few problematic outcomes, and in production-level systems they become fat-tailed critical risks that have outsized consequences.

=== "Problematic consequences of out-of-distribution data"
    1. **False positive & False Negative**: prediction does not match the ground truth.
    2. **Unreliable True positive & True Negative**: prediction matches the ground truth but due to samples being out-of-distribution, the performance of these predictions becomes very erratic. 

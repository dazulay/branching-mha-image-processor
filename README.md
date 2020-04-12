# Branching Image Processor with Aggregating MHA Encoders
Multi-Head Attention Neural Network for Image Datasets with Multiple Classification Tasks
This fully functioning notebook implements an architecture for processing an image dataset which contains multiple classification tasks.

Each batch of images is passed through an ImageToEntities module which creates entitities representing convolutional output features.  The samples in batch are then split by classification task, and passed though a parallel set of MHA Encoders -- one per classification task in the dataset. This is similar to the image-handling approach in ["Relational Deep Reinforcement Learning"](https://arxiv.org/abs/1806.01830).

Each of these AggregatingMHAEncoders consists of N stacks of MHA/Normalization/Feed-Forward layers, based upon the encoder portion of the encoder/decoder architecture originally described in ["Attention is All You Need"](https://arxiv.org/abs/1706.03762).  A final layer in each AggregatingMHAEncoder -- either a Max Pooling function, or "AggegatedMHA" function -- reduces the dimensionality of each encoders output.  

The sub-batches output by the encoders is then re-combined into a single batch for a final module, with each sample in the batch concatenated with it's associated classification task, and this recombined batch is then passed to a final feed-forward layer.

For testing, this repository contains one dataset from ["An Explicitly Relational Neural Network Architecture"](https://arxiv.org/abs/1905.10307). 

Each of these architectural elements are shown in more detail in diagrams in the rest of the notebook.


<img src="images/branching_mha_encoder.png">

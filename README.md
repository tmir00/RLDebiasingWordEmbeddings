# Debiasing Word Embeddings: A Reinforcement Learning Approach

## Setup

**Note: RUN THIS FIRST**
- Section Dependencies (What should be run first): None 

This section is responsible for installing all dependencies, importing all required packages and initializing important words.


## Bias Analysis Helper Functions

**Note: RUN THIS**
- Section Dependencies: Setup

The Bias Analysis Helper Functions visualize and quantify bias within word embeddings:  

1. The `plot_gender_direction_projection` function visualizes how professional titles are projected onto a gendered axis, revealing their associative biases with gendered terms. 
   
2. The `generate_cosine_similarity_heatmap` creates heatmaps to show the closeness of words to gendered reference points, highlighting potential biases. 
   
3. The `analyze_gender_bias_in_embeddings` function statistically examines the skew of professional titles along the gender dimension, incorporating tests like Shapiro-Wilk for normality and Mann-Whitney U for distribution comparisons. 


## Embeddings Analysis (OPTIONAL)

- Section Dependencies (What should be run first): Setup
- File Dependencies (What embeddings are needed): `glove.twitter.27B.200d.txt`, `GoogleNews-vectors-negative300.bin`, `cc.en.300.bin`
- ***File path updates required for Load Data code blocks*** 

This section consisted of our initial deep dive into bias within embeddings. We conduct a comprehensive examination of various word embeddings, specifically GloVe, Word2Vec, and FastText, to understand potential gender biases. For each embeddings, we employ a normalization-loaded function `load_<embedding_type>_embeddings` to parse and vectors from a pre-trained embeddings file. We then identify existing professional titles within these embeddings, establish a 'gender direction' by contrasting male and female-associated vectors. With this, we visualize the bias by projecting professional titles onto the gender direction and creating heatmaps to illustrate the cosine similarity between these titles and gendered terms. 

## Algorithm: Data Processing

**Note: RUN THIS**
- Section Dependencies (What should be run first): Setup, Bias Analysis Helper Functions.
- File Dependencies (What embeddings are needed): `geneval-sentences-both-dev.en_ar.en`
- ***File path update required for `Counter Factual Analysis` block.

**Algorithm: Data Processing**

The Data Processing stage involves three key phases:

**Data Analysis:**
A sample dataset from the Common Crawl is loaded. This text file is then used to train a FastText model.

**Soft Debiasing:**
Soft debiasing is applied directly to the word embeddings. This technique involves reducing the projection of word vectors onto the gender direction, a quantified representation of gender bias in the embedding space. Visualizations of projections and cosine similarity heatmaps before and after debiasing illustrate the effect of this process.

**Counterfactual Data Analysis:**
Counterfactual data analysis involves training a new FastText model on a different text corpus with counterfactual examples. This section includes visualizations and bias analysis.

## Algorithm

**Note: RUN THIS**
- Section Dependencies (What should be run first): Setup, Bias Analysis Helper Functions, Algorithm: Data Processing.
- File Dependencies (What embeddings are needed): None

This section of the project is where the heart of the bias mitigation process lies. It's structured into several key components.

**Action Helper Functions:**
This subsection includes functions essential for the debiasing actions.  `calculate_gender_direction` computes vector representations and identify bias directions in the embedding space. `calculate_single_embedding_bias` quantifies the bias of individual embeddings, while `calculate_semantic_similarity` measures how closely the debiased word embeddings resemble the original, ensuring that meaning is retained. Lastly, `neutralize` and `apply_debiasing_action` actively modify word embeddings to mitigate bias, and `incremental_adjustment_cda` gradually shifts biased embeddings toward less biased counterparts using Counterfactual Data Augmentation (CDA) principles.

**Environment:**
Here we define a custom Gym environment, `EmbeddingDebiasingEnv`, which frames the debiasing task as a reinforcement learning problem. The actions, states, and rewards are defined here to navigate the trade-off between reducing bias and preserving semantic content.

**Run Algorithm:**
In this phase, a Deep Q-Network (DQN) agent is trained within the previously defined environment. The agent explores different debiasing actions to learn an optimal strategy for bias mitigation. By interacting with the environment through numerous episodes, the model accumulates knowledge on the effects of various actions, seeking to maximize the rewards that represent successful debiasing.

**Result Analysis:**
Once the algorithm has run, the bias and semantic similarity of the resulting embeddings are visualized and analyzed. The `plot_gender_direction_projection` function visualizes the debiased embeddings, revealing their new positions relative to the gender direction. `generate_cosine_similarity_heatmap` offers a heatmap of cosine similarities, providing a detailed look at the semantic changes post-debiasing. Lastly, `analyze_gender_bias_in_embeddings` uses statistical methods to offer a final assessment of bias in the debiased embeddings, ensuring the algorithm's effectiveness.

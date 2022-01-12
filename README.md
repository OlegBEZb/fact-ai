# fact-ai
A repository for the FACT-AI course at the UvA. Contains the reproduction of the implementation of our chosen paper.

# TODO-list
1. Download one of the dataset
2. Preprocessing (Yuri)
   1. we lowercase all text, 
   2. parse and obtain POS tags using spaCy (Honnibal et al., 2020), 
   3. tokenize the text into unigrams, 
   4. and filter words that occur fewer than 10 times in the training dataset.
3. Implement the tools for similarity evaluation (René)
   1. PCA
   2. WEAT
4. Get the 100-d embeddings from word2vec (René)
5. Optional
   1. Train the model for each of the datasets (Oleg)
   2. Mimic the plot styles
   3. Check the situation for the recent models like transformers and so on
   4. Expand the analysis with other biases like age, citizenship, occupation, orientation
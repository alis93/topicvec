# TopicVec
TopicVec is the source code for "Generative Topic Embedding: a Continuous Representation of Documents" (ACL 2016).

This is a modified fork of the original TopicVec repository available at https://www.google.co.uk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjs0qrBxsjVAhVG2hoKHSx9C3gQFggoMAA&url=https%3A%2F%2Fgithub.com%2Faskerlee%2Ftopicvec&usg=AFQjCNEFQqOILJOhgVEwMPMS_6V3eALgwA

The modifications are roughly as follows:
* Options are loaded by a configuration file rather than hardcoded. Check config.yml for options that can be changed.
* Removed all unnecessary code. In the original, the authors included code for experimenting against many different models. They have been removed and code reduced to only that required for running the topicvec algorithm.
*Palmetto has been included, for better evaluation of topics.
*Other minor changes.

#Inputs

Corpus file is the original corpus of documents that will be used. 
The corpus is loaded line by line. Each line is treated as a single document and is tokenized into sentences and then tokenized into words by the corpusloader file. 

vocab file is the vocabulary, consisting of unique words.
This should be a single word on each line.

The default top1grams-wiki.txt also contains probabilities and frequencies of words, by using wikipedia. The probabilities are used in setting the priors
For a custom vocab file, a list of words without probabilities or frequencies will be fine. The algorithm will use default priors instead.

The word vec file contains the word embeddings. 
The first line should contain the number of word embeddings and the length of each embedding(dimensionality)
The format of the embeddings should be as follows:
	word embedding_vector

for example if there are 5 words that have been embedded into vectors with length 10:
	5 10
	hello -9.4104 3.0343 -0.0234 2.4545 3.0343 -0.0234 2.4545
	This 2.6323 3.0343 -0.0234 2.4545 3.0343 -0.0234 3.3545
	is -4.1023 3.0343 -0.0234 2.4545 3.0343 -0.0234 1.4d45
	some 1.5077 3.0343 -0.0234 2.4545 3.0343 -0.0234 0.4545
	embedding 0.5033 3.0343 -0.0234 2.4545 3.0343 -0.0234 2.4545


#Outputs

The algorithm will use the embedding file and vocabulary to generate a numpy file.
The numpy file contains 4 items.
*The embeddings
*The words
*Dict of word to index in embedding
*skipped words

A log file is generated, which output the results of each iteration, including the topics.

Embeddings of the top topics are generated in -best-topic.vec
Embeddings of the last topics are generated in -best-topic.vec

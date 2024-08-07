---
comments: true
---

# Embeddings

In this tutorial, we will be covering embeddings leveraging on Ollama using `ollama pull mxbai-embed-large`.

This is a critical prerequisite as it is the foundation of the "R" of Retrieval Augmented Generation (RAG).

After this tutorial, we can move on to the "AG" of RAG which focuses on leveraging on LLMs to take our top retrieved results and answer questions.

In totality, we will cover a range of basic to advanced RAG architectures spanning across:

1. semantic similarity through vector search using biencoders,
2. lexical similarity through keyword search,
3. ranking most relevant results through semantic reranking using cross-encoders,
4. semantic chunking
5. fine-tuning LLMs using QLoRA and LoRA, and
6. a combination of all of the above.

## Environment Setup

!!! tip "Options"

    === "**Easy Option | Install Ollama Directly**"

        If you do not want to run apptainer, feel free to just install the [Ollama](https://ollama.com) executable, and you can get up and running with all the Ollama commands like `ollama serve` and `ollama pull mxbai-embed-large` for example.

    === "**Hard/Scalable Option | Using Apptainer**"

        Follow our [tutorial on Apptainer](https://www.deeplearningwizard.com/language_model/containers/hpc_containers_apptainer/) to get started. Once you have followed the tutorial till the [Ollama section](https://www.deeplearningwizard.com/language_model/containers/hpc_containers_apptainer/#ollama-multi-modal-workloads-example-llava7b-v16) where you successfully ran `ollama serve` and `ollama pull mxbai-embed-large`, you can run the `apptainer shell --nv --nvccli apptainer_container_0.1.sif` command followed by `jupyter lab` to access and run this notebook.
        
        When you shell into the Apptainer `.sif` container, you will need to navigate the directory as you normally would into the Deep Learning Wizard repository that you cloned, requiring you to `cd ..` to go back a few directories and finally reaching the right folder. 

## Embedding Example

In this section, we will cover the basics of embeddings which in simple terms, given a bunch of words, we can represent them by a bunch of numbers using an embedding model.


```python
# Import the Ollama module
import ollama
```

Here, we use a simple sentence "An apple a day is good for you" to illustrate embeddings. In the following code we convert a sample sentence into sentence embeddings.


```python
# Create a sample sentence
text = 'An Apple a day is good for you!'

# Pass the sentence to our embedding model 
embeds = ollama.embeddings(model='mxbai-embed-large', 
                           prompt=f'{text}')
```


```python
# The object returned is a dictionary
type(embeds)
```




    dict




```python
# With the key 'embedding'
embeds.keys()
```




    dict_keys(['embedding'])




```python
# We can access the value through the key 'embedding' which returns a list
type(embeds['embedding'])
```




    list



We can see the sentence is represented by a vector of numbers totally a count of 1024, this varies, and can be a larger or smaller number.


```python
# Check length of list
len(embeds['embedding'])
```




    1024




```python
# Sample 10 elements in the list
embeds['embedding'][:10]
```




    [-0.6146689653396606,
     0.22128450870513916,
     0.19337491691112518,
     -0.6306122541427612,
     -0.7143896818161011,
     0.3647981584072113,
     0.05146953463554382,
     0.5699551105499268,
     0.8753001093864441,
     0.983851432800293]



If you look at the first 10 of our embedding of the sentence, you can see it's a bunch of positive/negative floating numbers that represent our sentence. Doing this allows us to convert text to a fixed dimension of numbers (in a vector) and run meaningful mathematical operations on them such as cosine similarity to have a simple indication of similarity of sentences/words.

## Embedding Comparisons

In this section, we:

1. create 3 sentence,
2. leverage on the `mxbai-embed-large` embedding model, we convert these 3 sentences individually into sentence embeddings essentially converting 3 bunch of texts into 3 vectors of numbers.
3. run cosine similarity amongst the sentence embeddings to measure the similarity amongst the sentences. 

There are many different measures beyond cosine similarity like L1 distance (Manhattan distance), L2 distance (Euclidean distance), dot product similarity, max inner product similarity, and many more. However, cosine similarity is one of the most common and we chose it as regardless of the different methods to measure similarity, the other algorithms achieve the same objective of measuring similarity.

### Cosine Similarity


```python
# Text & Embeddings 1
text = 'Apple, oranges, and grapes are good fruits.'
embeds_1 = ollama.embeddings(model='mxbai-embed-large', 
                             prompt=f'{text}')

# Text & Embeddings 2
text = 'Eating a good balance of meat, vegetables, and fruits everyday is good for you.'
embeds_2 = ollama.embeddings(model='mxbai-embed-large', 
                             prompt=f'{text}')


# Text & Embeddings 3
text = 'How to be a good data engineer?'
embeds_3 = ollama.embeddings(model='mxbai-embed-large', 
                             prompt=f'{text}')
```

The cosine similarity between two vectors $ \mathbf{A} $ and $ \mathbf{B} $ is calculated using the following formula:

$$
\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|}
$$

Where:

- $\mathbf{A} \cdot \mathbf{B}$ denotes the dot product of vectors $\mathbf{A}$ and $\mathbf{B}$.
- $\|\mathbf{A}\|$ denotes the Euclidean norm (magnitude) of vector $\mathbf{A}$.
- $\|\mathbf{B}\|$ denotes the Euclidean norm (magnitude) of vector $\mathbf{B}$.

The dot product of two vectors is the sum of the products of their corresponding components. Mathematically, if $\mathbf{A} = [a_1, a_2, ..., a_n]$ and $\mathbf{B} = [b_1, b_2, ..., b_n]$, then the dot product $\mathbf{A} \cdot \mathbf{B}$ is:

$$
\mathbf{A} \cdot \mathbf{B} = a_1 \times b_1 + a_2 \times b_2 + \cdots + a_n \times b_n
$$

The Euclidean norm of a vector is the square root of the sum of the squares of its components. For a vector $\mathbf{V} = [v_1, v_2, ..., v_n]$, the Euclidean norm $\|\mathbf{V}\|$ is:

$$
\|\mathbf{V}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

So, to calculate the cosine similarity between two vectors, we take their dot product and divide it by the product of their magnitudes.

This formula computes the cosine of the angle between the two vectors. If the angle is small (cosine close to 1), the vectors are similar; if it's large (cosine close to -1), they are dissimilar. If the angle is 90 degrees (cosine 0), the vectors are orthogonal and have no similarity.



```python
import numpy as np

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity
```

Above, we created a cosine similarity function using numpy, and this can be easily replicated in PyTorch, Jax, and any other libraries that optimise vector computations.

#### Cosine Similarity: Sentence 1 vs Sentence 3

Given the sentences `Apple, oranges, and grapes are good fruits.` and `Eating a good balance of meat, vegetables, and fruits everyday is good for you.` we should expect some level of similarity, and in turn, a higher score. Cosine similarity returns a floating number that ranges between 0 to 1, the closer to 1 indicates higher similarity. And we can see it's indeed high at 0.70.


```python
# Comparison 1/2
cosine_similarity(np.array(embeds_1['embedding']), np.array(embeds_2['embedding']))
```




    0.6990337492754806



#### Cosine Similarity: Sentence 1 vs Sentence 3

Given the sentences `Apple, oranges, and grapes are good fruits.` and `How to be a good data engineer?` we should expect a low level of similarity, and in turn, a lower score. Indeed, in this case we can see how it is 0.30 which is substantially lower than 0.70 when we ran cosine similarity between sentence 1 and sentence 2.


```python
# Comparison 1/3
cosine_similarity(np.array(embeds_1['embedding']), np.array(embeds_3['embedding']))
```




    0.31111347652792015



## Summary

With a basic understanding of embeddings and how we convert sentences into sentence embeddings, we can now proceed to cover the "R" of RAG, which represents "Retrieval" followed by the "AG" of RAG which leverages on LLMs to take our retrieved chunks and answers questions.

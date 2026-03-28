# [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)

Existing Systems says that all of words are treated as atomic units, no similarity sense between words.
good reasons: simplicity, robustness & fact that simple model on huge data outperform complex model on less data.

but they are limited: amount of in-domain data for speech recognition is limited
there are situations where simple scaling up of the basic techniques will not result in
any significant progress, and we have to focus on more advanced techniques.

* **Goals**
introduce techniques to learn high quality word vectors from huge dataset(size: billion words)
similar words will be close to each other with a degree of similarity(can have multiple degrees of similarity)
similarity of word representations goes beyond simple syntactic regularities.

for example:  Using a word offset technique where simple algebraic operations are performed on the word vectors, it was shown for example that vector(”King”) - vector(”Man”) + vector(”Woman”) results in a vector that is closest to the vector representation of the word Queen.

* **Previous Work**
neural network language model (NNLM) was proposed: feedforward neural network with a linear projection layer and a non-linear hidden layer was used to learn jointly the word vector representation and a statistical language model.

neural network(1 hidden layer)= word vectors trained
nnlm= trained on top of word vectors
architectures were significantly more computationally expensive for training

## **Model Architectures**

use Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) for estimation of words
we focus on distributed representations of words learned by neural networks cause they work better than LSA

For all the following models, the training complexity is proportional to 
$$[O = E × T × Q]$$


E - no of training epochs
T - no of words in training set 

* **Feedforward Neural Net Language Model (NNLM)**
At the input layer, N previous words are encoded using 1-of-V coding, where V is size of the vocabulary
The input layer is then projected to a projection layer P that has dimensionality N × D, using a shared projection matrix

becomes complex as values in the projection layer are dense.
computational complexity per each training example : $$Q = N × D + N × D × H + H × V$$      {dominating term is H × V}

N = 10; 
size of the projection layer (P) might be 500 to 2000 ;
hidden layer size H = 500 to 1000 units ;

most of the complexity is caused by the term N × D × H.

use hierarchical softmax where the vocabulary is represented as a Huffman binary tree.


* **Recurrent Neural Net Language Model (RNNLM)**
proposed to overcome certain limitations of the feedforward NNLM, 
such as the need to specify the context length (the order of the model N),

It does not have a projection layer; only input, hidden and output layer. 

 complexity per training example of the RNN model is
 $$Q = H × H + H × V$$

H × V can be efficiently reduced to H × log2(V ) by using hierarchical softmax.

* **Parallel Training of Neural Networks**
use distributed framework called DistBelief; allows to run multiple replicas of the same model in
parallel, and each replica synchronizes its gradient updates through a centralized server that keeps
all the parameters

use mini-batch asynchronous gradient descent with an adaptive learning rate procedure called Adagrad

* **New Log-linear Models**

We propose two new log-linear models (CBOW and Skip-gram) that eliminate the non-linear hidden layer, reducing complexity from O(H×V) to O(V), making training on billions of words feasible. The key insight is that by removing the hidden layer, we can learn word vectors more efficiently while still capturing syntactic and semantic relationships through algebraic operations on the learned vectors.


## **Continuous Bag-of-Words (CBOW)**

CBOW predicts the **current word** using the **context (surrounding words)**.

Key idea:
Instead of modeling full word order like NNLM, CBOW treats context as a **bag of words**, meaning **word order does not matter**. All context word embeddings are projected into the same vector space and **averaged (or summed)**.

Architecture simplification compared to NNLM:

* removes the non-linear hidden layer
* shares projection matrix across all words
* averages context vectors → predicts center word
* uses log-linear classifier

Example window (size = 2):

context = { w(t−2), w(t−1), w(t+1), w(t+2) }
predict = w(t)

Training objective:
maximize probability of the correct center word given surrounding context.

Training complexity per training example:

$$
Q = N \times D + D \times \log_2(V)
$$

Where:
N = number of context words
D = dimensionality of word vectors
V = vocabulary size

Insights:

* averaging context embeddings smooths representation
* very efficient training
* works well for frequent words
* captures semantic similarity
* ignores word order → loses some syntactic precision


## **Continuous Skip-gram Model**

Skip-gram does the **inverse task of CBOW**.

Instead of predicting center word from context, Skip-gram:
predicts **context words from the current word**.

Example window (size = 2):

input = w(t)

predict:
{ w(t−2), w(t−1), w(t+1), w(t+2) }

Skip-gram tries to maximize probability of observing surrounding words given the center word.

Intuition:
a word should be useful for predicting words appearing near it.

Training complexity:

$$Q = C \times (D + D \times \log_2(V))$$

Where:
C = maximum distance of context window

Insights:

* works better for rare words
* captures fine semantic relationships
* slightly slower than CBOW
* more training signal per word
* distant words are sampled less frequently (because less related)

## **Dual Vector Representation**

To calculate the probability $P(w_{t+j} \mid w_t; \theta)$, Word2Vec uses **two vectors per word**:

**Two Vectors per Word $w$:**

1. **$v_w$** (center vector): used when $w$ is a **center word** (the input/target word)

2. **$u_w$** (context vector): used when $w$ is a **context word** (appearing in surrounding window)

### **Probability Calculation**

For a center word $c$ and a context word $o$:

$$P(o \mid c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

This is a **softmax function** over the entire vocabulary $V$.

**Components:**

- **Numerator**: $\exp(u_o^T v_c)$ measures compatibility/similarity between context word $o$ and center word $c$

- **Denominator**: $\sum_{w \in V} \exp(u_w^T v_c)$ normalization term ensuring probabilities sum to 1

- **Dot product**: $u_o^T v_c$ captures how likely word $o$ appears near word $c$

**Why Two Vectors?**

- Separates roles: predicting vs being predicted

- Makes optimization more efficient  

- Provides flexibility in modeling asymmetric relationships

- Final word embeddings typically use only $v_w$ (center vectors) or average both

**Training Process:**

1. For each position $t$ in the corpus

2. Use center vector $v_{w_t}$ for the current word

3. Maximize probability of context words using their context vectors $u_{w_{t+j}}$

4. Update both $v$ and $u$ vectors via gradient descent (typically SGD or Adagrad)

**Computational Challenge:**

The denominator requires summing over the entire vocabulary $V$ (often 10,000 - 100,000+ words):

$$\sum_{w \in V} \exp(u_w^T v_c)$$

This is computationally expensive.

**Solutions:**

1. **Hierarchical Softmax**: Reduces complexity from $O(V)$ to $O(\log_2 V)$ by organizing vocabulary as a Huffman binary tree

2. **Negative Sampling**: Approximate by sampling a few negative examples instead of computing full softmax


---

## **CBOW vs Skip-gram**

| Aspect           | CBOW          | Skip-gram            |
| ---------------- | ------------- | -------------------- |
| predicts         | center word   | surrounding words    |
| input            | context words | single word          |
| speed            | faster        | slower               |
| rare words       | weaker        | stronger             |
| semantic quality | good          | very good            |
| training signal  | averaged      | multiple predictions |
| complexity       | lower         | slightly higher      |

rule of thumb:
CBOW → fast training on large corpora
Skip-gram → better embeddings for semantic tasks

---

## **Word Relationships via Vector Arithmetic**

Word embeddings capture **linear relationships** between concepts.

Example:

$\vec{X} = \vec{biggest} - \vec{big} + \vec{small}$

Result:
closest vector to **smallest**

classic example:

king − man + woman ≈ queen

Why this works:
embedding space encodes directions representing relationships:

gender direction
tense direction
plural direction
capital-city relation

Examples of relationships captured:

semantic:
country → capital
France : Paris :: Germany : Berlin

syntactic:
run → running
big → bigger → biggest

man → woman

---

## **Evaluation: Semantic-Syntactic Relationship Test Set**

Word vectors are evaluated using analogy questions of form:

A : B :: C : ?

example:
Athens : Greece :: Oslo : Norway

Types of relationships tested:

semantic relations:

* capital city
* currency
* city-state
* family relations (brother-sister)

syntactic relations:

* adjective → adverb (apparent → apparently)
* comparative (great → greater)
* superlative (easy → easiest)
* verb tense (walk → walked)
* plural nouns (mouse → mice)

Dataset stats:

* 8869 semantic questions
* 10675 syntactic questions
* only single-token words used (no "New York")

---

## **Results & Observations**

Training data:
Google News corpus (~6 billion tokens)

Key observations:

1. increasing vector dimension improves accuracy up to a point
2. increasing training data improves performance significantly
3. diminishing returns appear after certain dimensionality
4. Skip-gram performs best on semantic tasks
5. CBOW performs well on syntactic tasks and trains faster

Example results (300-d vectors):

Skip-gram:
semantic accuracy ≈ 50%
syntactic accuracy ≈ 55%

CBOW:
semantic accuracy ≈ 15%
syntactic accuracy ≈ 53%

Training time comparison:

CBOW:
~1 day

Skip-gram:
~3 days

---

## **Key Insight of Paper**

removing hidden layer → massive speedup
hierarchical softmax → reduces computation from V to log₂(V)
simple models + huge data > complex models + small data

log-linear architectures (CBOW, Skip-gram) scale efficiently to billions of tokens and still capture:

semantic structure
syntactic structure
analogical reasoning capability


## mental model

- **CBOW**:
ignores order of context words (bag-of-words assumption).
context → meaning of word
predict missing word from surroundings

context words are projected into embeddings and averaged, then the model predicts the most probable center word.
> words that appear in similar contexts should have similar vectors

$P(w_t \mid w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2})$

```
__ cat sat on __ mat
      ↓
predict "the"
```

given the situation, what word fits here?
maximize probability of correct center word given surrounding bag-of-words


treats context as an unordered set.
key idea:
order information is discarded, but semantic signal is preserved because nearby words strongly constrain meaning.

example:

```
eat food now
consume meal today
```

similar contexts → similar embeddings



- **Skip-gram**:
word → environment it appears in
predicts multiple context words from a single center word.

meaning emerges from statistical co-occurrence structure of language.
predict surroundings from word


so bag-of-words will truly ignore the order of words and well tries to keep the predicted word in center, well not predicted word, correct term will be closest accurate word 

continuous skip gram will try to get multiple context words from current words
`the small cat sat on the mat`

`small` -> `{the, cat}`
`
small → the
small → cat
`
words appearing in similar contexts have similar vectors.

given this word, what situations does it appear in?



word → context

$$P(w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2} \mid w_t)$$

> a word is defined by the environments it appears in
> words appearing in similar contexts will have similar embeddings.



example:

```
small dog
small house
small problem
```

and

```
tiny dog
tiny house
tiny problem
```

→ vectors of **small** and **tiny** become close.


 why relationships emerge (important insight)

Skip-gram does **not explicitly learn grammar rules**, but patterns appear because contexts are structured.

example contexts:

```
big dog
bigger dog
small dog
smaller dog
```

so relationship direction becomes consistent:

$v_{bigger} - v_{big} \approx v_{smaller} - v_{small}$

this leads to analogy solving:

king − man + woman ≈ queen

[implementation](./word2vec.py)

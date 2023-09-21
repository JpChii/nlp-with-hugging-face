# Making Transformers Efficient in Production

In previous notebooks, we've seen how transformers can be fine-tuned for various tasks. However in many situations irrespective of the metric, model is not very useful if it's too slow or too large to meet the buisness requirements of the application. The obvious alternative is to train a faster and more compact model, but with reduction comes the degradation in performance. What if we need a compact yet highly accurate model?

In this notebook we'll cover four techniques with Open Neural Network Exchange(ONNX) and ONNX Runtime(ORT) to reduce the prediction time and memory footprint of the transformers, they are:

  1. *Knowledge distillization*
  2. *Quantization*
  3. *Pruning*
  4. *Graph Optimization*

We'll also see how the techniques can be combined to produce significant performance gains. [How Roblox engineering team scaled Bert to serve 1+ Billion Daily Requests on CPUs](https://medium.com/@quocnle/how-we-scaled-bert-to-serve-1-billion-daily-requests-on-cpus-d99be090db26) and found that knowledge distillization and quantization improved the latency and throughput of their BERT classifier over a factor of 30!

To illustrate the benefits and tradeoffs associated with each technique, we'll use intent detection(important component of text-based assistants), where low latency is critical for maintaining a conversation in real time.
Along the way, we'll also learn how to create custom trainers and perform hyperparamter search, and gain a sense of what it takes to implement cutting-edge research with Transformers(lib).

## Creating a Performance Benchmark

Like other machine learning models, deploying transformers in production environments involves trade-off among several constraints, the most common being:

*Model Performance*

How well does our model perform on a well-crafted test set that reflects production data? This is especially important when the cost of making erros is large(and best mitigated with a human in the loop) or running inference on millions of examples and small improvements to the model metrics can translate into large gains in aggregate.

*Latency*

How fast can our model deliver predictions? We usually care about latency in a real-time environment where we deail with a lot of traffic, like how stack overfloww needed a classifier to quickly [detect unwelcome comments on the website](https://stackoverflow.blog/2020/04/09/the-unfriendly-robot-automatically-flagging-unwelcoming-comments/)

*Memory*

How can we deploy billion-parameter models like GPT or T5 which requires gigabytes of disk storage and RAM? Memory plays an important role in mobile and edge devices, where we've to generate predictions without a cloud server.

Failing to address these constrains might result in:

* Poor user experience
* Balooned cloud costs for just a few user requests

To explore how the 4 different compression techniques can be used to optimizer these. Let's create a benchmark class which measures each of these quantities for a given pipeline and a test set.

## Making Models Smaller Via Knowledge Distillation

Knowledge Distilliation is a general-purpose technique to train a smaller *student* to mimic the larger *teacher* model. This was introduced in 2006 for ensemble models then popularized in 2015 for deep learning and applied to image classification and speech recognition.

With the increase in parameters with pretrained models(trillions and more), Knowledge distillation is a compression technique to compress these models and build practical applications.

### Knowledge Distillation for Fine-Tuning

#### Intution
How is knowledge distilled or transferred from teacher to student. In Fine-tuning the main idea is to augment(extend) the logits to soft probabalities. With soft probabalities we'll get probabality distribution(ex higher probablity for two intents) or information that is not accessible from labels alone. By training the student to mimic these probabalities we distill the information to the student.

#### Mathametical perspective

* We feed an input sequence X to the teacher to generate a vector of logits z(x) = [z_1(x), ..., z_N(x)]. We can convert these logits to probabalities with softmax function:

$\text{softmax}(z(x)) = \frac{\exp(z_i(x))}{\sum_{j=1}^N \exp(z_j(x))}$

* But with this softmax, we'll mostly get a single highest probability for an intent and others close to zero. With this the student can'e learn anything from truth labels.
* To soften the probabalities we'll use the hyperparameter T before applying the softmax

${p_i(x)} = \frac{\exp\left(\frac{z_i(x)}{T}\right)}{\sum_{j=1}^N \exp\left(\frac{z_j(x)}{T}\right)}$

when T=1 we recover the original softmax distribution.

*hard-vs-soft softmax distribution*

![alt](../notes/images/8-Making-transformers-efficient-in-production/hard-vs-soft-softmax.png?raw=1)

What's the difference between teacher and student softened probabalies?

* We can calculate this with [Kullback Leibler (KL)](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) divergence to measure the differnce between two probabalities.

$D_{\text{KL}}(p, q) = \sum_i p_i(x) \log \left(\frac{p_i(x)}{q_i(x)}\right)$

The divergence increase when the differnce in loss is large.

* We can calculate how much is lost when we approximate the probabality distribution of studfen with the teacher. This is knowledge Distillation loss. Here we include $T^2$ because before softmax we divide the logits by T in numerator and denominator of softmax. Product of $T^2$ and KL divergence gives the loss.

$L_{\text{KD}} = T^2 \cdot D_{\text{KL}}$

* $T^2$ is the normalization factor to account for the magnitude of the gradients produced by soft label scales as 1/$T^2$.
* For classification tasks, the student loss is then a weighted average of the distillation loss with the usual cross-entropy loss $L_{\text{CE}}$ of the ground truth labels. We use a hyperparameter $\alpha$ to control control the relative strength of the losses.

$L_{\text{student}} = \alpha L_{\text{CE}} + (1 - \alpha) L_{\text{KD}}$

*Knowledge distillation process*


![alt](../notes/images/8-Making-transformers-efficient-in-production/entire-distillation-process.png?raw=1)

### Knowledge Distillation for Pretraining

Knowledge distillation can also be used during pretraining to create a general-purpose student that can be subsequently fine-tuned on downstream tasks. In this case Masked Language Modelling Knowledge of BERT is transferred to the student. For example, in [DistilBERT paper](https://arxiv.org/abs/1910.01108) mlm loss is augmented with a term from knowledge distillation and a cosine embedding loss to align the directions of the hidden state vectors between the teacher and student.

$L_{\text{cos}} = 1 - \cos(h_s, h_t)$

$L_{\text{DistilBERT}} = \alpha L_{\text{mlm}} + \beta L_{\text{KD}} + \gamma L_{\text{cos}}$

We already have a fine-tuned BERT model, let's use knowledge distillation to fine-tune a smaller and faster model. For this we;ve to augment the cross entropy loss with $L_{\text{KD}}$. We can do this by creating our own trainer.

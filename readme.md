# AI in 2023: Skills and Concepts

## Table of Contents

- [Modeling using Deep Neural Networks in PyTorch](#modeling-using-deep-neural-networks-in-pytorch)
- [Bonus: Shap / XGBoost](#bonus-shap--xgboost)
- [Training, Classification, and Decision Trees: Tic-Tac-Toe](#training-classification-and-decision-trees-tic-tac-toe)
- [Convolutional Neural Nets and Transfer Learning: Image Classification](#convolutional-neural-nets-and-transfer-learning-image-classification)
- [Generative Adversarial Networks (DCGAN): Image Generator I](#generative-adversarial-networks-dcgan-image-generator-i)
- [Embeddings: Dense Vector Representations of Semantic Information](#embeddings-dense-vector-representations-of-semantic-information)
- [Transformers I: Overview of Self-Attention](#transformers-i-overview-of-self-attention)
- [Transformers II: Explore LLaMa Source Code](#transformers-ii-explore-llama-source-code)

## Links

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Hugging Face](https://huggingface.co/)
- [Papers with Code](https://paperswithcode.com/sota)
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction.html)
- [SHAP Docs](https://shap.readthedocs.io/en/latest/index.html)
- [MarkTechPost (AI news/papers)](https://www.marktechpost.com/)

## Modeling using Deep Neural Networks in PyTorch

### Code

[Lesson](energy/energy-lesson.py)
[Reference](energy/energy-ref.py)

### Links

[Install Pytorch](https://pytorch.org/get-started/locally/)

### Assignments

- Use own dataset
- Challenge: Optimize lr
- Challenge: Optimize architecture

### Vocab

- Training: Adjusting the parameters of a neural net from training data.
- Inference: Running a trained neural net to get an output for real.

- Linear or Fully Connected Layer: A trainable layer that connects every input to output with linear weights and a bias per output. Weights are a matrix.
- Activation Layer: Non-linear layer. Needed sandwiched between linear layers to model complex behaviour. ReLu is the cheapest and simplest.


- Scalar: An individual value. Think a floating point value.
- Vector: A bunch of values. Like x and y together form a position vector.
- Matrix: Drawn on paper in two dimensions. Could represent a set of equations or every combination of vectors. The topic of Linear Algebra has a lot to say about matrices. Modern AI hardware does large matrix multiplication at a blazing rate.
- Tensor: A higher level data structure that can be any of the above to the nth degree. A Rank 0 tensor is a scalar, rank 1 is a vector, rank 2 is a matrix, and so forth. The shape of a tensor describes it's rank and the dimensions at each level. Tensor is the data type in PyTorch that unifies all behaviour and allows data to be sent to a GPU. Think "Everything is a Tensor."

- Weights: A matrix that has values to multiply by every input to get every output.
- Biases: A vector that is an offset to add to each output of a linear layer.
- Parameters: The weights and biases of the all the linear layers of a model taken together.

- Loss Function (Cost Function): A function that takes the actual outputs (labels) and predicted outputs (from the forward pass) for each training sample and determines how far off it was. This is the starting point for adjusting the weights for each data point in training. CrossEntropyLoss is often used for classification.
- Mean Squared Error Loss: Simple linear regression without taking the sqrt.
- Loss: The actual numerical value that is calculated using the loss function with the actual and predicted output vectors as arguments.

- Epoch: Running through all the training data once. 

- Data Set (PyTorch): Manages actually loading a complete data set and any data augmentations.
- Data Loader (PyTorch): Inner training loop should use this iterator. Manages mini-batches on hardware including shuffling, batch size, and floating point precision. Returns a tuple of inputs and labelled outputs.

### Notes

- Setup: `pip install torch matlibplot scikit-learn pandas`


## Bonus: Shap / XGBoost

### Code

[Shap / XGBoost Lesson](energy/bonus/energy-shap-xgboost.py)

### Links

[Main Lesson Source](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%C2%A0insights.html)

[XGBoost Background](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

[Machine Vison Example](https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.html)

### Assignments
- Analyze the model you created using your own data using SHAP
- Re-train your model using XGBoost
- Challenge: Identify any confounding features in your model and find any cases where using your model would cause problems because of these

### Vocab

- Explainability
- Principle Component Analysis

## Training, Classification, and Decision Trees: Tic-Tac-Toe

### Code

[Tic Tac Toe](tictactoe/tictactoe.py)

### Links

- [SGD Visualization](https://aurifexlabs.com/ai-tutorials/sgd/sgd.html)

### Assignments

- Modify the architecture and retrain until you have a minimal number of parameters while still having zero loss. Try changing the input and/or output encoding.
- Challenge: Train a model to predict the best next move

### Vocab
- Gradients: The partial derivative of loss with respect to each weight. Think of the input, labeled output, and predicted output (calculated in the forward pass) as constant for each training sample, and each gradient represents how much changing a single weight would affect the loss (or improve the model for that particular sample).PyTorch (Autograd) is able to take the derivatives automatically from a Python implementation of the forward pass.
- Chain Rule: From multivariable calculus, it basically says that you can multiply the gradients from each layer to get  the next one. Or in other words, how much a change to the weight in a lower layer affects the final output (and loss) is the product of all those relationships through all the intermediate layers.
- Backpropagation: The process of recording the gradients at each layer and using them to calculate the gradients in the previous layer using the Chain Rule. PyTorch does the heavy lifting behind the scenes with just a few details in the training script.
- Learning Rate: The value that determines how much to adjust the weights during every pass. Often controlled by a more complex function with it's own hyperparameters.
- Hyperparameter: A value like learning rate that is tuned to improve the training process. Not a parameter in the model itself. Usually a global variable in the training script or in a .json config file.

- Stochastic: A system with randomness. See non-determinism and random variables.
- Gradient Descent: Using the backpropagated gradients, we actually adjust the weight in the opposite direction of the gradient to go in the direction (in n-dimensional space) "most downward" in the sense that it minimizes loss. Remember that the gradient represents how an increase in the weight would increase the loss, and we are trying to reduce the loss so we multiply the gradient by -1; or descent. The landscape that we are going down is specific to that data sample, but by choosing an appropriate learning rate and dropout, we can get to a stable (and good enough even if not the best) minimum where loss is low and accuracy is high for a real problem.
- Dropout: Randomly selecting only some of the weights when adjusting the weights during training. Dropout increases robustness and reduces overfitting.
- Local Minimum: For a multivariate function if we calculate the gradient and iteratively go down this, we'll hit a point where all the partial derivatives become zero; equalibrium. But there may be a point lower somewhere far away, and the most low of them all is the global minimum. But another wrinkle is that in practice we are iterating through different samples that all have totally different landscapes, so we always have to remember we are trying to find a stable local minimum that's good enough and that is not overfit to a single data point.

- Overfitting: A model that it overly specific to a narrow set of training data. Small data sets lead to overfitting generally.


- Entropy: A measure of disorder of a system. Or think of it as the inverse of the ability to predict something about the system. See Information Theory. 
- Cross Entropy Loss: Also known as log loss or logistic loss. In this case loss uses the ratio of entropy which invloves a log.

- Logits: The probability scores output by a model. It can be thought of a one hot encoding of the classification or prediction. Usually this is then put through a final classifier layer.

- Softmax: The most common classifier. It works like an exponentially scaled probability. It's so effective because it still results in output that is usable in non-deterministic (high gain) contexts because the probabilities are scaled so that only alternatives that are almost as good are considered.

- Accuracy: Percentage of the samples where the prediction is correct.

## Convolutional Neural Nets and Transfer Learning: Image Classification

### Code

[PyTorch Tutorial: Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

### Links

- [Transfer Learning Notes](https://cs231n.github.io/transfer-learning/)

### Assignments

- Try transfer learning on your own images
- Challenge: Compare ResNet, SqueezeNet, and one other model from [Papers with Code](https://paperswithcode.com/sota)

### Vocab

- Transfer Learning: Additional training on a pre-trained model
- Frozen Parameters: Weights that aren't adjusted when doing additional training
- Fixed Feature Extracture: Only retraining the final fully connected layer (only retraining the classifier)
- Fine Tuning: Adjusting the weights of a model during additional training
- Kernel / Filter / Convolution: Layer with learnable parameters that represent values of a set of image processing kernels. Generally the image dimensions will get smaller and more feature 
- Feature Map: The image being processed that represents features of the input data in the middle of the model. The input is an image, the output is a class, and everything in the middle is a feature map
- Residual Connections: Bypass connections that allow lower level information to bypass learning layers during the forward pass and gradients to likewise flow and help learning
- Pooling Layers: Fized max or averaging processing filter that generally reduces an image's size


## Generative Adversarial Networks (DCGAN): Image Generator I

### Code

[PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
[Datasets](dcgan/datasets/)

### Links

[Goodfellow's Paper](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)

### Assignments

- Use your own images
- Try SGD instead. What happened?
- Challenge: Hyperparameter tuning of Adam optimizer / Try Lion optimizer
- Challenge: Try ResNet for the discriminator

### Vocab

- Generator: A model that generates an output. In the DCGAN case the output is an image. The input is a latent vector that represents the space of all possible outputs.
- Discriminator: A model that tells whether a generated output is real or fake
- Minimax: A two player game technique in which each player minimizes their opponents maximum score at each point in the game
- Binary Cross Entropy Loss: CrossEntropyLoss for one output neuron instead of two for a binary classification
- Adam Optimizer: An adaptive optimizer (also see Lion optimizer)
- Convolutional Transpose Layer: Sometimes called a deconvolution. Opposite of a convolution layer so that the generater can generate images from a latent vector instead of generating a feature from an image.
- Batch Normalization Layer: Normalize values per mini-batch. Can be used instead of dropout.

## Embeddings: Dense Vector Representations of Semantic Information

### Code

[Word Embeddings: Encoding Lexical Semantics](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html?highlight=embeddings)

### Links

### Assignments

- Train on different source text
- Challenge: Try adjusting the number of dimensions and use SHAP to try to understand what they represent

### Vocab
- N-Gram: The vector of n tokens used to generate simple correlations (compared to self-attention). The n in n-gram and the context length in transformers are the same thing.
- Tokenizer: A program that splits our input text into tokens before it can be encoded using an embedding. Can be a traditional code/NLP tokenizer.


## Transformers I: Overview of Self-Attention

### Code

### Links

- [Transformer Diagrams](https://jalammar.github.io/illustrated-transformer/)
- [GPT-2 Specific Diagrams](https://jalammar.github.io/illustrated-gpt2/
)
- [Supplimentary Explaination](https://machinelearningmastery.com/the-transformer-attention-mechanism/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

### Assignments
- Use BERT on HuggingFace

### Vocab
- Self-Attention: input -> Q,K, V matrices -> dot product Q and transpose of K -> softmax (into scores) -> score V -> output
- Dot Product: Product of magnitudes and cos of angle between (in n space); how close and big are two vectors
- Multi-Head: multiple in parallel
- Feed Forward: MLP (Linear + Activation layers)
- Positional Encoder: Encodes the relative (or absolute) position of a token. Needed because all tokens are effectively parallel and need explicit positional information.


## Transformers II: Explore LLaMa Source Code

### Code

[LLaMa Source Code](https://github.com/facebookresearch/llama/blob/main/llama/model.py)

### Links

[LLaMa Paper](https://arxiv.org/abs/2302.13971)
[OpenLLaMA on HuggingFace](https://huggingface.co/openlm-research/open_llama_3b)
[MPT-7B Details](https://www.mosaicml.com/blog/mpt-7b)
[MPT-7B on HuggingFace](https://huggingface.co/mosaicml/mpt-7b)

### Assignments

- Run inference using OpenLLaMa-3B locally using HuggingFace
- Challenge: Compare LLaMA-7B and MPT-7B on a benchmark or for a hand written test

### Vocab



## LLM Explainability with SHAP

## Fine-Tuning LLMs in the Cloud: Training Parallelism

## Prompt Engineering and Few/Zero Shot Learning

## DCGAN II

## Stable Diffusion

## Reinforcement Learning I

## Reinforcement Learning II

## LangChain and Autonomous Agent

## Multi-Modal: Audio/Video

## LLMs with Long Context Length: mpt-7b-storywriter

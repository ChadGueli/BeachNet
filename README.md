# BeachNet

This repository is a Jupyter notebook containing modules that I created in 2020 for the [M5 - Uncertainty](https://www.kaggle.com/c/m5-forecasting-uncertainty) challenge, but never submitted. While reading it please keep in mind that it was written before the cosine annealing LR schedule was published, and it is written in TF2.3, or maybe 2.4, who remembers?

The M5-Uncertainty challenge, like the previous M\* challenges were an attempt to further the theory of time series modeling. This particular challenge focused on predicting the distribution of an estimate. Participants had to predict the distributions of hundreds of products 30 days into the future. The data was provided by Walmart.

The predictions were evaluated using the [pinball loss](https://www.lokad.com/pinball-loss-function-definition) of 9 quantiles from the predicted distributions. 

## Idea
My solution involves using the ideas from [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) and deconvolving at the end of each block to take a time series with Nx1 input and mapping it to a 30x9 output.

Before going any further, it is worth noting that I believe drop out would provide unsuitable estimates of the quantiles. For those unfamiliar, as noted in Gal and Ghahramani's canonical [2015 paper](https://arxiv.org/pdf/1506.02142.pdf), we may interpret a neural net with dropout as an instance of a deep Gaussian process under the Bayesian paradigm. However, each time series is an individual observation with dependent components, so there is no long run aspect to reasonably rely upon for a Gaussian approximation.

One might also wonder: Why not just output 9 filters, one for each quantile, and call it a day? When using a conv net, we do not model the relationship between filters, but rather assume that information represented across filters will be combined as needed at a higher level. In practice this works, but in theory, particularly at the output layer of a network, it is tantamount to assuming that the filters are independent. Since the quantiles of a distribution are dependent, I attempted to model the relationship between them. I regret not applying a monotonicity constraint in the quantile dimension. (Honestly, at the time I didn't even know it was feasible).

## Implementation
I began by implementing a 2D convolution that was causal in the time dimension but not in the quantile dimension. This was simply a matter of not padding the "front" of the data; i.e. time N+i for i an integer in \[1, 10\].

Then, I created a causal separable 2D convolution. This is similar to the causal 2D convolution, but obviously, separable.

With the individual layers constructed, I put these pieces together to create a module similar to the MBConv, and dubbed this atrous, casual, "mobile" convolution the BeachModule.

NOTE: Context refers to other information, e.g. product, that describes each time series.

The code is provided as is, the neural architecture search, model object, and training have been removed, but feel free to implement your own versions. Also there are a lot of artifacts from my data storage choices.

## Pinball Loss Commentary

The following is a musing I had in May of 2022, no research was conducted for this segment. If you know of an improvement, please let me know.

In quantile regression, the goal is to predict the kth quantile. The pinball loss achieves this by defining the loss surface such that the minima is achieved when the estimator splits the data such that k% of the sample lies below the estimate. This competition made use of the scaled pinball loss, so the loss was divided by the L1 norm of the differenced series. The pinball loss is a valuable construct, but it begins to break down when we only see part of the data. Obviously, we can look at the validation performance, but this hides a more sinister issue.

Since the model divides the data at the kth quantile, we can reframe the task as a classification problem; i.e. classify values as belonging to at least the kth quantile. On this interpretation, class is latent with the time series values as the corresponding observation. This creates an extremely tricky dependence structure with class at time N+i dependent on the X values at time N+i, which are themselves dependent on the preceding X values. To make matters worse, quantile extremity and class imbalance are positively correlated. While these issues are challenging, they affect all models optimized with this loss, even those looking at all of the data at once.

Of course, when training a neural network, we use mini batches. So, not only is class imbalance more pronounced because at each time step we limit our scope to small portions of the data, but we have now inadvertently included the mini batch in the dependence structure. In theory, given enough epochs, the batch effect will disappear. But, I imagine the time horizon will also depend on quantile extremity. In any case, despite its flaws, the pinball loss is a reasonable loss function, especially given its task: predicting uncertainty.

My proposal: Use a form of boosting at the end of each epoch to reweight the observations in a way that reflects their quantile. For those concerned about storing 10x the data, a positional encoding could be used to represent an observations quantile, this value could then be transformed to get the weight. This boosting proposal should accelerate elimination of the batch effect, and help account for class imbalance.

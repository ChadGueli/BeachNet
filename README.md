# BeachNet

This repo is a Jupyter notebook containing modules that I created in 2020 for the [M5 - Uncertainty](https://www.kaggle.com/c/m5-forecasting-uncertainty) challenge, but never submitted.

Like the previous M\* challenges, the M5-Uncertainty challenge was an attempt to further the theory of time series modeling.

This particular challenge focused on predicting the distribution of an estimate, participants had to predict the distributions of hundreds of products 30 days into the future. The data was provided by Walmart.

The predictions were evaluated using the [pinball loss](https://www.lokad.com/pinball-loss-function-definition) of 9 quantiles from the predicted distributions.

While reading this notebook, please keep in mind it is written in TF2.3 before support for specifying padding in each direction was added.

## Idea
My solution involves using the ideas from [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) and deconvolving at the end of each block to take a time series with Nx1 input and map it to a 30x9 output.

Before going any further, it is worth noting that dropout would have been inappropriate as it allows estimation of uncertainty about the estimate, not the target. 

Additionally, I decided against an output sequence with 9 filters because it enabled smaller kernels. For example, applying a 1D conv of size 3 to K features for each of Q quantiles in a tensor of size KQxNx1 (CxWxH) would take a kernel of size 3KQN. Where a 2D conv of size 3x3 applied to a KxNxQ tensor would require a kernel only of size 9KN.

## Implementation
I began by implementing a 2D convolution that was causal in the time dimension but not in the quantile dimension. This was simply a matter of not padding the "front" of the data; i.e. the next time steps.

Then, I created a causal separable 2D convolution. This is similar to the causal 2D convolution, but obviously, separable.

With the individual layers constructed, I put these pieces together to create a module similar to the MBConv, and dubbed this atrous, casual, "mobile" convolution the BeachModule.

NOTE: Context refers to other information, e.g. product, that describes each time series.

The code is provided as is, the neural architecture search, model object, and training have been removed, but feel free to implement your own versions. Also there are a lot of artifacts from my data storage choices.

## Things I'd Do Differently
Remove the deconvolutions because they lead to checkerboard patterns, as discussed by Odena, et. al. [(2016)](https://distill.pub/2016/deconv-checkerboard/).

Start my NAS with a lot of layers to initially process the series in its 1D form.

Use some bidirectional feature pyramid network layers, as proposed by Tan, et. al. [(2019)](https://arxiv.org/pdf/1911.09070.pdf), with fixed length and depth, but variable width.

Finally, as advocated in Meng, et. al. [(2022)](https://arxiv.org/pdf/2203.12683.pdf), I would remove the atrous convolutions because they are slow to compute.

## Pinball Loss Commentary

The following is a musing I had in May of 2022, no research was conducted for this segment. If you know of an improvement, or see any tortured phrases, please let me know.

In quantile regression, the goal is to predict the qth quantile. The pinball loss achieves this by defining the loss surface such that the minima is achieved when the estimator splits the data such that p% of the sample lies below the estimate, where p=F(q) and F is our cdf. This competition made use of the scaled pinball loss, so the loss was divided by the L1 norm of the differenced series. The pinball loss is a valuable construct, but it begins to break down when we only see part of the data. Obviously, we can look at the validation performance, but this hides a more sinister issue.

Since the model divides the data at the qth quantile, we can reframe the task as a classification problem; i.e. classify values as belonging to at least the qth quantile. On this interpretation, class is latent with the time series values as the corresponding observation. This creates an extremely tricky dependence structure with class at time N+i dependent on the X values at time N+i, which are themselves dependent on the preceding X values. Or class dependent on y dependent on X. To make matters worse, quantile extremity and class imbalance are positively correlated. While these issues are challenging, they affect all models optimized with this loss, even those looking at all of the data at once.

Of course, when training a neural network, we use mini batches. So, not only is class imbalance more pronounced because at each time step we limit our scope to small portions of the data, but we have now inadvertently included the mini batch in the dependence structure. In theory, given enough epochs, the batch effect will disappear. But, I imagine the time horizon will also depend on quantile extremity. In any case, despite its flaws, the pinball loss is a reasonable loss function, especially given its task: predicting uncertainty.

My proposal: Use a form of boosting at the end of each epoch to reweight the observations in a way that reflects their quantile. For those concerned about storing 10x the data, a positional encoding could be used to represent an observations quantile, this value could then be transformed to get the weight. This boosting proposal should accelerate elimination of the batch effect, and help account for class imbalance.

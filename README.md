# BeachNet

This repository is a Jupyter notebook containing the modules that I created in 2012 and never submitted to the [M5 - Uncertainty](https://www.kaggle.com/c/m5-forecasting-uncertainty) challenge.

The M5-Uncertainty challenge, like the previous M\* challenges was an attempt to further the theory of time series modeling. This particular challenge focused on predicting the distribution of an estimate. Participants had to predict 9 quantiles of the distribution 30 days into the future for hundreds of products. The data was provided by Walmart.

My solution involves using the ideas from [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) and deconvolving at the end of each layer to take a time series with Nx1 input and mapping it to a 30x9 output. Doing this required implementing a lot of things.

No, I didn't feel that drop out would provide suitable estimates of the quantiles. My reasoning is simple, each time series is an individual observation with dependent components, so there is no long run anything to reasonably rely upon for a Gaussian approximation.

Firstly, I had to implement a 2D convolution that was causal in the time dimension but not in the quantile dimension. This was simply a matter of not padding the "front" of the data.

Then, I created a causal seperable 2D convolution. This is similar to the causal 2D convolution, but obviously, seperable.

With the individual layers constructed, I put these pieces together to create a module similar to the MBConv, and dubbed this atrous, casual, "modile" convolution the BeachModule.

NOTE: Context refers to other information, e.g. product, that described each time series.

The code is provided as is, the neural architecture search, model object, and training have been removed, but feel free to implement your own versions. Also there are a lot of artifacts from my data storage choices. Finally, it was written for TF2.4, or maybe 2.3, who remembers? Enjoy!

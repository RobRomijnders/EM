# Expectation Maximization
This repo implements and visualizes the Expectation maximization algorithm for fitting Gaussian Mixture Models.
We aim to visualize the different steps in the EM algorithm. Bishop provides a great explanation in his book _pattern recognition and machine learning (Bishop, 2006)
The entire code for the EM is less than 100 lines._

The lines in the code refer to the corresponding equations in the book. We focus on equations 9.12 up to 9.40

# EM as an alternative to gradient descent for non-convex objectives
Probably, there's many visualizations like these on the web. However, we focus on visualizing the optimization that EM does. For models with unobserved or hidden variables, the log likelihood contains multiple modes. Therefore, good old _gradient descent_ will likely get stuck in a local minima. EM performs way better on this optimization task by using some nice tricks.

# Local approximation
One can consider the E-step as being a local approximation to the log likelihood of the data. We refer to this as the expected log likelihood. Interpret this for Gaussian mixtures as follows: if we knew which cluster is responsible for which data, we could calculate the parameters of the mixtures. Conversely, if we knew the parameters of the mixtures, we could calculate the cluster responsibilities. It turns out the E-step takes care of the former, the M-step of the latter. 
In the M-step, we optimize the mixture parameters using the expected sufficient statistics for the hiddem variables. Hence the name, **expected** log likelihood (see the explanation surrounding eq.9.40 in Bishop). This E-step can be interpreted as a local approximation of the (complete) log likelihood. This local approximation is convex. For GMM, it even has a closed form solution.

# Experiment
The visualizations aim to show the effect of locally approximating the log likelihood. 
  * We generate three clusters. Similar to figure 9.5 in (Bishop,2006)
    * One can specify their own parameters in lines 15-18
  * We perform EM with the object _fitGMM()_
  * In the _util.py_ we have many helper functions for visualizing the mixtures

# Results
  * The red lines plot the complete data log likelihood. We see this function is non-convex. Sometimes, we also observe the local maxima next to the global maximum
  * The blue lines plot the expected log likelihood. This can be interpreted a local approximation to the log likelihood. We observe that it is convex

 ![gif1](https://github.com/RobRomijnders/EM/blob/master/im/EM_cherry4.gif?raw=true)
 ![gif2](https://github.com/RobRomijnders/EM/blob/master/im/EM_cherry3.gif?raw=true)
 ![gif3](https://github.com/RobRomijnders/EM/blob/master/im/EM_cherry1.gif?raw=true)

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com



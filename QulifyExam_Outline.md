## Revised Proposed Extensions

### 1. Modeling

* Integrate the _uncertainty of wind speed_ into the formulation;
* Consider numbers of drones with different sizes as part of the decision process, which may requires to _change the objective function_;
* (__IF POSSIBLE__) Consider the _fairness issue_ into the formulation;

### 2. Solution Methodology

   After reformulate the stochastic programming (SP) model, the following things will be done:

* Directly solve the formulation using Gurobi solver with _default settings_, a poor performance is expected;
* Adapt the scenarios decomposition algorithm (SDA) for the new formulation, and _solve the new SP model using SDA_;
* Reformulate the new formulation using sample average approximation (SAA), adapt an _integer Benders decomposition algorithm (i-BDA)_ for it, and solve the SAA model with i-BDA;

  Both __numerical results__ and __intuitions__ for _using SDA_ and _SAA via Benders_ will be explained, including  __required assumptions__ to use these two methods.

### 3. Analysis and Discussion

* Evaluate the value of stochastic solution (VSS) and Expected Value of Perfect Information (EVPI);
* Analyze the differences between the results of different formulations;
* (__IF POSSIBLE__) Analyze how would including the fairness change the results of experiments;
* Discuss the empirical meaning of our extensions: compared to the model proposed in the original paper, what are the influences of the new settings? Does the experiments' result matches the expectations?

### 4. Potential Concerns

My main concerns to this project is my access to the data. In the original paper, the authors used some "fancy" methods to generate the intensities of the earthquake in different areas, and further they leveraged those values to estimated how severe were the road damaged. I doubt if I can access the reality data, and do the similar sophisticated assessment in my qualifying exam. 

My current thought is to generate some data of randomness in the model using Python packages under a given probability distribution, for simplicity. 

As for the second suggestion proposed by Dr. Terlaky, I think reformulate the original problem using the robust optimization (RO) method would be a good solution to it. However, RO is in a different frame with SP, which is also a new area to me. I will learn some knowledge of RO and try to explore the implementation of it onto our problem, but it is very possible that I will choose to finish it last.  


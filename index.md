# Project Title:
## Members:
- Ethan Bolton
- Carlo Casas
- Evan Chase
- Felix Pei
- Sagar Singhal

---

## Background and Problem Definition
Understanding how housing prices correlate with demographics can help to predict how property values will change with changes in the population, income, ethnicity, areas of employment, and other demographic factors of an area. This insight can help investors in real estate decide where to buy and sell property and could inform governments on how they can make housing more affordable.

This project aims to predict housing prices of counties in the United States given demographic data and discover relevant correlations using the American Community Survey of 2015 and the Zillow Rent Index, which tracks the median housing price per square foot of a given area. 

## Methods
Our problem is a regression problem, where we attempt to approximate the relationship between independent variables (population, etc.) and a dependent variable (median rent).

First, we need to preprocess the data, which includes both numerical and categorical variables. 
For numerical data, we should normalize the values, and for categorical data, we can try simple integer encoding or one-hot encoding depending on the variable.

Next, we need to learn a mapping from the input features to the output. There are many approaches we can choose from for this regression task. From linear approaches, we can experiment with simple linear regression, Lasso regression, and GLMs like Gamma regression. Among non-linear approaches, we can try support vector regression and feedforward neural networks, and we can also use additional input features generated from non-linear transformations of original inputs with the linear methods. Afterward, we can compare training time and performance of these various approaches, and we can probe the trained models to see what features are particularly informative of the output.

## Potential Results / Discussion
The regression analysis will yield a relationship between median rent of United States counties and the various demographics of each. Given the wide range of demographic data from the census dataset, we seek to find what parameter or set of parameters correlates to the highest or lowest rent prices. Examples of these demographic parameters include age, ethnicity, income, poverty, and unemployment, commute time, industry distribution, etc. While some parameters seem directly correlated, others may yield unexpected dependence to rent. 

## References
3 References

## Proposed Timeline
### Project Proposal (10-7)

The main idea of what our project - estimating rent based on the defining parameters of a County in the US - will consist of.

| Intro/Background | Problem Definition | Methods | Potential Results and Discussion | References | Timeline | Proposal Video |
| ---------------- | ------------------ | ------- | -------------------------------- | ---------- | -------- | -------------- |
| Sagar | Sagar | Felix | Evan | Sagar | Carlo | Ethan |


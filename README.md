# Time-Series-Challenge

In this 3-hour data Challenge, we are provided by dataset of the history of retail details (sales, purchase_price, quantities...) of some products.

Our goal is to train a Time-Series model on this data. Since no instruction on what the target variable is, we decided to proceed with a multivariate approach, where we design a TS model fitting all the features history simultaneously by crossing their values.

A quick data exploration enabled us to reach the following conclusions:

- There are 3 different products in the dataset. Since the behavior of the quantites vary considerably depending on the product, we decided to focus only. on one product.
- The size of the dataset is considerably small. After removing duplicates (the intial dataset was duplicated, and this adds no value to the model we eventually train). Therefore, for a single product, we are left with around 300 entires, which is small.
- Since the size of dataset is too small, any attempt to use Deep Learning Approaches (RNN, LSTMs...) to train this model will very likley underfit, and won't give good results. Therefore, we went with a simple Multivariate TS algorithm: Vector autoregression (VAR process), which provides a way of estimating relationships between the time series and their lagged values.
- Due to the time constraint, we trained only one model, and reported one metric RMSE. Had we had enough data, we would have validated the model using cross-validation techniques. 

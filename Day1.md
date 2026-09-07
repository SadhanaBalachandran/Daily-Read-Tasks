# In Machine learning What are metrics? Difference between metrics vs Loss

## What is a Loss Function?
A loss function, also known as a cost function or objective function, is used to measure how well the model’s predictions match the true values. During training, the loss function quantifies the error or discrepancy between the predicted values and the actual values, guiding the optimization process to improve the model's performance.

## Purpose of Loss Function:
**Optimization:** The primary purpose of the loss function is to provide a measure of how well the model is performing during training. The optimization algorithm, such as stochastic gradient descent (SGD) or Adam, uses this loss to adjust the weights of the model to minimize the error.

**Model Training:** By minimizing the loss function, the model parameters are adjusted in a way that improves the accuracy of the predictions.

*Examples of Loss Functions:*

*Mean Squared Error (MSE):* Commonly used for regression tasks. It calculates the average of the squared differences between predicted and actual values.

*Binary Crossentropy:* Used for binary classification tasks. It measures the performance of a classification model whose output is a probability value between 0 and 1.	

*Categorical Crossentropy:* Used for multi-class classification tasks. It measures the performance of a classification model whose output is a probability distribution across multiple classes.

## What is a Metric?
Metrics are used to evaluate the performance of the model during training and testing. Unlike loss functions, metrics are not used to optimize the model; instead, they provide additional information about the model's performance.

- Purpose of Metrics:
	- Evaluation: Metrics help in evaluating and monitoring the performance of a model by providing additional insights into the accuracy and effectiveness of predictions.
	- Monitoring: Metrics are often used to track how well the model is performing on various aspects during training and evaluation.

* Examples of Metrics: *
	- Accuracy: The proportion of correctly predicted instances out of the total instances. It is commonly used for classification tasks.
	- Precision and Recall: Precision measures the accuracy of positive predictions, while recall measures the ability to find all positive instances.
	- F1 Score: The harmonic mean of precision and recall, providing a single metric that balances both aspects.
	- Mean Absolute Error (MAE): Measures the average magnitude of errors in regression tasks, without considering their direction.

## Key Differences between Loss Function and Metric in Keras

# Here’s a table summarizing the differences between loss functions and metrics in Keras:

|Aspect|LossFunction|Metric|
|------|------------|------|   
Definition|Measures the error between predicted and actual values|Evaluates and monitors the performance of the model|
|Primary Use|Optimization: Adjusts model parameters during training|Evaluation: Provides additional performance insights|
|Impact on Training|Directly impacts the training process by guiding optimization|Does not affect the training process directly
|Usage in Compilation|Specified in the compile() method to train the model|Specified in the compile() method for performance monitoring|
|Purpose|To minimize error and improve model accuracy|To assess various aspects of model performance|
|ExamplesMean Squared Error (MSE), Binary Crossentropy, Categorical Crossentropy|Accuracy, Precision, Recall, F1 Score, Mean Absolute Error (MAE)|


## Conclusion
Loss functions and metrics are both crucial in the model training and evaluation process in Keras. While the loss function is essential for optimizing the model, metrics provide additional insights into the model’s performance. Understanding their roles helps in effectively designing and assessing machine learning models.

# Class Imbalance & SMOTE

## 1. What is Class Imbalance?

**Class imbalance** occurs in a classification problem when one class has significantly more observations than another class.

For example, imagine we are building a model to detect fraudulent transactions:

| Class     | Transactions |
| --------- | -----------: |
| Not Fraud |        9,500 |
| Fraud     |          500 |

Here:

* **Not Fraud** = 95%
* **Fraud** = 5%

The **Not Fraud** class is the **majority class**.
The **Fraud** class is the **minority class**.

This is called **class imbalance**.

### Visual representation

```text
Not Fraud  ███████████████████████████████████████████████  95%
Fraud      ███                                               5%
```

The important point is that the classes do **not** have to be exactly 50–50 for imbalance to exist. The severity depends on the dataset and the problem.

---

# 2. Why is Class Imbalance a Problem?

A model trained on imbalanced data may become biased toward the majority class.

Consider this situation:

```text
Actual data:

Not Fraud → 9,500
Fraud     →   500
```

Suppose a model predicts:

> **Every transaction is Not Fraud.**

The model would correctly classify 9,500 transactions.

Therefore:

```text
Accuracy = 9,500 / 10,000
         = 95%
```

That sounds like a very good model.

But the model detected:

```text
Fraud detected = 0 / 500
```

So the model completely failed to identify the minority class.

This is why **accuracy alone can be misleading when classes are highly imbalanced**.

---

# 3. What Metrics Should We Look At?

When dealing with class imbalance, we should look beyond accuracy.

Important metrics include:

* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**
* **ROC-AUC**
* **PR-AUC**, especially when the positive class is very rare

The appropriate metric depends on the business problem.

For example, in fraud detection, missing a fraudulent transaction may be much more costly than incorrectly flagging a legitimate transaction.

So understanding the business objective is important when deciding which metric to prioritize.

---

# 4. How Can Class Imbalance Be Handled?

There are several approaches.

### Common approaches

```text
                 CLASS IMBALANCE
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Resampling     Class Weights   Threshold
        │                              Tuning
   ┌────┴────┐
   ↓         ↓
Oversampling Undersampling
```

### Main techniques

| Technique            | What it does                                           |
| -------------------- | ------------------------------------------------------ |
| **Oversampling**     | Increases the number of minority-class observations    |
| **Undersampling**    | Reduces the number of majority-class observations      |
| **SMOTE**            | Creates synthetic minority-class observations          |
| **Class weights**    | Gives more importance to the minority class            |
| **Threshold tuning** | Changes the probability cutoff used for classification |

This document focuses mainly on **SMOTE**.

---

# 5. What is SMOTE?

**SMOTE** stands for:

> **Synthetic Minority Over-sampling Technique**

SMOTE is a technique used to address class imbalance by generating **synthetic samples for the minority class**.

The most important word here is:

### Synthetic

SMOTE does not simply duplicate existing minority observations.

Instead, it creates **new artificial observations based on existing minority observations and their nearest neighbors**.

---

# 6. Why Not Simply Duplicate the Minority Class?

Suppose our original data contains:

```text
Not Churn → 9,000
Churn     → 1,000
```

One simple approach would be to duplicate the Churn observations.

For example:

```text
Original:

A
B
C


After simple oversampling:

A
A
A
B
B
B
C
C
C
```

We are simply repeating the same observations.

SMOTE takes a different approach.

It creates new observations based on the existing minority observations.

```text
Original minority observations:

A ●
        ● B
             ● C


SMOTE:

A ●────● New sample
        \
         ● New sample
             \
              ● C
```

The new points are **synthetic observations**.

---

# 7. How Does SMOTE Create Synthetic Samples?

Let's use a simple example.

Suppose two minority-class customers have:

| Customer | Age | Monthly Bill |
| -------- | --: | -----------: |
| A        |  30 |      ₹40,000 |
| B        |  40 |      ₹60,000 |

SMOTE identifies that A and B are relatively close minority-class observations.

It can create a synthetic observation somewhere between them.

For example:

| Customer      |    Age | Monthly Bill |
| ------------- | -----: | -----------: |
| A             |     30 |      ₹40,000 |
| **Synthetic** | **35** |  **₹50,000** |
| B             |     40 |      ₹60,000 |

The synthetic customer did **not** exist in the original dataset.

It was generated from the characteristics of existing minority observations.

---

# 8. Does SMOTE Simply Take the Average?

Not exactly.

This is an important distinction.

SMOTE generally:

1. Selects a minority-class observation.
2. Finds its **nearest minority-class neighbors**.
3. Selects one of those neighbors.
4. Creates a new point somewhere between the two observations.

Conceptually:

```text
Minority observation A
        ●
         \
          \
           ●  ← Synthetic observation
            \
             \
              ●
        Minority observation B
```

The synthetic observation is created along the line between the existing observations.

The exact location involves a random value.

---

# 9. The Basic Mathematical Idea

Suppose one feature has:

```text
A = 30
B = 40
```

The difference is:

```text
40 - 30 = 10
```

SMOTE can choose a random value between 0 and 1.

Suppose:

```text
Random value = 0.4
```

Then:

```text
New value = 30 + 0.4 × (40 - 30)

          = 30 + 4

          = 34
```

So the synthetic value could be:

```text
Age = 34
```

The same concept is applied across the relevant numerical features.

In practice, the SMOTE library performs these calculations for us.

---

# 10. What Does "Nearest Neighbor" Mean?

SMOTE needs to know which minority observations are similar or close to one another.

It therefore uses the concept of **nearest neighbors**.

Imagine minority observations represented as points:

```text
             B ●

     A ●             C ●

             D ●
```

For observation A, SMOTE looks for nearby minority observations.

Those observations are its **nearest neighbors**.

SMOTE then uses these neighboring minority observations to create synthetic samples.

Therefore, the basic idea is:

> **Find similar minority observations → create new observations between them.**

---

# 11. What Does `k_neighbors` Mean?

When using SMOTE, we may see:

```python
SMOTE(k_neighbors=5)
```

`k_neighbors` tells SMOTE how many nearby minority-class observations to consider when generating synthetic samples.

For example:

```python
SMOTE(k_neighbors=5)
```

means that SMOTE considers the **5 nearest minority-class neighbors**.

The default value is commonly 5.

For a beginner, the important thing to remember is:

> **`k_neighbors` = number of nearby minority observations considered by SMOTE.**

---

# 12. Does SMOTE Always Make the Dataset 50–50?

No.

SMOTE does **not** require the classes to become exactly equal.

For example, suppose we start with:

```text
Majority class = 9,000
Minority class = 1,000
```

We could oversample the minority class to:

```text
Majority class = 9,000
Minority class = 5,000
```

There is no rule saying it must become:

```text
9,000 vs 9,000
```

The desired amount of oversampling depends on the problem.

The goal is to improve the model's ability to learn the minority class, not simply to make the class counts identical.

---

# 13. What is `random_state` in SMOTE?

SMOTE involves randomness when selecting neighbors and generating synthetic samples.

We can control this using:

```python
SMOTE(random_state=42)
```

`random_state` is used primarily for **reproducibility**.

It means:

> "Use the same random process every time I run this code."

For example:

```python
smote = SMOTE(random_state=42)
```

If we run the same process again with the same data and settings, we can reproduce the same result.

### Why 42?

There is nothing mathematically special about 42.

These are all valid:

```python
SMOTE(random_state=42)
SMOTE(random_state=10)
SMOTE(random_state=100)
```

42 is simply a commonly used convention.

### Important:

`random_state` is generally **not something we optimize to find the "best" value**.

Think of it as:

> **Random state = reproducibility, not performance tuning.**

---

# 14. Where Should SMOTE Be Applied?

This is one of the most important practical rules.

### ❌ Incorrect approach

Do not apply SMOTE to the entire dataset before splitting:

```text
Original Dataset
       ↓
     SMOTE
       ↓
Train/Test Split
```

This can cause **data leakage**.

---

# 15. What is Data Leakage Here?

The test set is supposed to represent **unseen data**.

If we apply SMOTE before the train-test split, information from the original dataset can influence the synthetic observations that eventually end up in the test set.

This can make the evaluation overly optimistic.

---

# 16. Correct SMOTE Workflow

Instead:

```text
Original Dataset
       ↓
Train / Test Split
       │
       ├───────────────┐
       ↓               ↓
   Training           Test
       ↓               ↓
     SMOTE          Leave it
       ↓             untouched
Balanced Training
       ↓
   Train Model
       ↓
Predict on Test
       ↓
Evaluate
```

### The rule:

> **Apply SMOTE only to the training data. Keep the test data untouched.**

This allows the test set to represent the real-world class distribution.

---

# 17. Complete Python Example

Let's build a small customer churn example.

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE
```

---

## Step 1 — Create the dataset

```python
data = {
    "Age": [25, 30, 35, 40, 45, 50, 28, 32, 38, 42],
    "MonthlyBill": [500, 600, 700, 800, 900, 1000, 550, 650, 750, 850],
    "Churn": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
}

df = pd.DataFrame(data)
```

Here:

```text
0 = No Churn
1 = Churn
```

Our dataset is intentionally imbalanced:

```text
No Churn → 8
Churn    → 2
```

---

## Step 2 — Separate Features and Target

```python
X = df[["Age", "MonthlyBill"]]

y = df["Churn"]
```

Remember:

```text
X → Features used for prediction

y → Target we want to predict
```

So:

```text
X
├── Age
└── MonthlyBill

y
└── Churn
```

---

## Step 3 — Split the Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

We split the data into:

```text
80% → Training data
20% → Test data
```

### Why `stratify=y`?

Because our target is imbalanced.

`stratify=y` attempts to preserve the class proportions when creating the train and test sets.

---

# 18. Apply SMOTE

Now we apply SMOTE **only to the training data**.

```python
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

The important line is:

```python
smote.fit_resample(X_train, y_train)
```

It means:

> Learn the minority-class structure from the training data and generate additional synthetic samples.

---

# 19. Check the Class Distribution

Before SMOTE:

```python
print(y_train.value_counts())
```

After SMOTE:

```python
print(y_train_resampled.value_counts())
```

You might see something similar to:

```text
Before SMOTE:

0    6
1    2
```

After SMOTE:

```text
0    6
1    6
```

The exact numbers depend on how the train-test split is produced.

The important point is that the minority class has been increased.

---

# 20. Train the Model

Now train the model using the resampled training data:

```python
model = LogisticRegression()

model.fit(
    X_train_resampled,
    y_train_resampled
)
```

Notice:

```text
SMOTE training data
        ↓
    Model training
```

We are **not** training the model on the untouched imbalanced training data.

---

# 21. Predict on the Test Data

Now use the original test data:

```python
y_pred = model.predict(X_test)
```

Notice the difference:

```text
Training data → SMOTE applied

Test data → SMOTE NOT applied
```

This is intentional.

---

# 22. Evaluate the Model

```python
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
```

The classification report provides metrics such as:

```text
Precision
Recall
F1-score
Support
```

These metrics help us understand how well the model is identifying each class.

---

# 23. Complete Code

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE


# 1. Create data
data = {
    "Age": [25, 30, 35, 40, 45, 50, 28, 32, 38, 42],
    "MonthlyBill": [500, 600, 700, 800, 900, 1000, 550, 650, 750, 850],
    "Churn": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
}

df = pd.DataFrame(data)


# 2. Separate features and target
X = df[["Age", "MonthlyBill"]]
y = df["Churn"]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Apply SMOTE only to training data
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)


# 5. Train the model
model = LogisticRegression()

model.fit(
    X_train_resampled,
    y_train_resampled
)


# 6. Make predictions on untouched test data
y_pred = model.predict(X_test)


# 7. Evaluate the model
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
```

---

# 24. When Can SMOTE Cause Problems?

SMOTE is useful, but it is not automatically the right solution for every dataset.

### 1. Overlapping classes

If minority and majority observations are heavily mixed together, SMOTE can sometimes generate synthetic points in regions where the classes overlap.

```text
Class 0 ● ● ●
        ●   ●
          ● ● Class 1
```

Synthetic samples in overlapping regions may make classification more difficult.

---

### 2. Minority-class outliers

If a minority observation is an outlier:

```text
● ● ● ●


                    ●
              Minority outlier
```

SMOTE may use that observation when generating synthetic samples.

This can introduce less useful synthetic observations.

---

### 3. Categorical features

Standard SMOTE is primarily designed for numerical feature spaces.

If your dataset contains categorical variables, techniques such as **SMOTENC** may be more appropriate.

---

### 4. Feature scaling and distance

SMOTE relies on distances between observations.

Therefore, the scale of numerical features can matter.

For example:

```text
Age       → 20–70
Income    → 20,000–200,000
```

Income has a much larger numerical scale than Age.

In practical ML pipelines, preprocessing and SMOTE should be arranged carefully, especially when cross-validation is involved.

---

# 25. SMOTE vs Other Approaches

| Method                  | Main idea                                                  |
| ----------------------- | ---------------------------------------------------------- |
| **Oversampling**        | Add more minority observations                             |
| **Random Oversampling** | Duplicate existing minority observations                   |
| **SMOTE**               | Generate synthetic minority observations                   |
| **Undersampling**       | Remove some majority observations                          |
| **Class Weighting**     | Give the minority class greater importance during training |
| **Threshold Tuning**    | Adjust the classification probability cutoff               |

SMOTE is therefore **one option**, not a mandatory step whenever class imbalance exists.

---

# 26. Important Interview Question

### Q: What is SMOTE?

A good Data Scientist answer:

> **SMOTE, or Synthetic Minority Over-sampling Technique, is a method for handling class imbalance by generating synthetic samples for the minority class. It creates these samples using existing minority observations and their nearest neighbors. SMOTE should generally be applied only to the training data to avoid data leakage.**

---

# 27. The SMOTE Mental Model

The easiest way to remember SMOTE is:

```text
MINORITY CLASS
      ↓
Find a minority observation
      ↓
Find nearby minority neighbors
      ↓
Choose a neighbor
      ↓
Create a point between them
      ↓
Synthetic minority observation
      ↓
Repeat as needed
```

### Three keywords:

**Minority class → Nearest neighbors → Synthetic samples**

---

# 28. Final Cheat Sheet

```text
CLASS IMBALANCE
│
├── One class has much fewer observations
│
├── Accuracy can be misleading
│
├── Important metrics:
│     ├── Precision
│     ├── Recall
│     ├── F1-score
│     └── Confusion Matrix
│
└── Possible solutions:
      ├── Oversampling
      ├── Undersampling
      ├── SMOTE
      ├── Class weights
      └── Threshold tuning
```

### SMOTE

```text
SMOTE
│
├── Synthetic Minority Over-sampling Technique
│
├── Works mainly with numerical feature spaces
│
├── Finds minority-class neighbors
│
├── Creates synthetic observations
│
├── Does NOT simply duplicate observations
│
├── Does NOT have to create a 50–50 dataset
│
├── random_state → reproducibility
│
└── Apply to TRAINING data only
```

---

# Key Takeaways

1. **Class imbalance** means the classes in a classification dataset are unevenly represented.

2. A model can have **high accuracy while performing poorly on the minority class**.

3. **SMOTE** stands for **Synthetic Minority Over-sampling Technique**.

4. SMOTE creates **synthetic minority-class observations** rather than simply copying existing observations.

5. It uses **nearest minority-class neighbors** to generate new samples.

6. `random_state=42` is used for **reproducibility**; 42 is not a special or "best" value.

7. SMOTE does **not necessarily need to make the classes 50–50**.

8. **Never apply SMOTE to the complete dataset before train-test splitting.**

9. The safe basic workflow is:

```text
Split
  ↓
SMOTE → Training data only
  ↓
Train model
  ↓
Predict on untouched test data
  ↓
Evaluate
```

10. SMOTE is a **tool for handling imbalance**, not a guarantee that the resulting model will perform better. Model performance should always be evaluated on appropriate, untouched validation/test data.

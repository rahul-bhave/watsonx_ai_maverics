## Purpose of solution:
This solution is built for the [IBM TechXchange Pre-Conference watsonx Hackathon](https://pretxchack.watsonx-challenge.ibm.com/), and we have selected the following path listed for the hackathon event.

---

## Path
Build a generative AI application for a use case supporting productivity with IBM watsonx.ai, featuring IBM Granite

---

# Use case: Customer Satisfaction Analysis using Watsonx AI

This project analyzes customer satisfaction based on feedback provided by customers who are using electronic devices from a specific manufacturer. The analysis utilizes IBM Watsonx AI and the Granite-13b-Instruct-V2 Foundation Model to perform sentiment analysis on customer feedback data. This repository contains the code, dataset, and necessary configurations to run the analysis.

---

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Project Setup](#project-setup)
- [Model Overview](#model-overview)
- [Data](#data)
- [Model Inference and Evaluation](#model-inference-and-evaluation)
- [Results](#results)
- [Productivity Improvement](#productivity-improvement)
- [Conclusion](#conclusion)

---

## Introduction

Customer satisfaction plays a vital role in understanding the success of a product and improving its overall experience. In this project, we aim to analyze customer reviews and classify whether the customers were satisfied based on their feedback. We utilize IBM's Granite-13b-Instruct-V2 model on Watsonx.ai to perform this task.

---

## Requirements

To run this project, the following packages are required:

- Python 3.x
- `datasets` library
- `scikit-learn==1.3.2`
- `ibm-watsonx-ai`

You can install these dependencies using the following commands:

```bash
pip install datasets
pip install "scikit-learn==1.3.2"
pip install -U ibm-watsonx-ai
```

---

## Project Setup

1. **IBM Watsonx Credentials**:
   Define your Watsonx credentials to interact with IBM's Foundation Models:

   ```python
   from ibm_watsonx_ai import Credentials
   import getpass

   credentials = Credentials(
       url="https://us-south.ml.cloud.ibm.com",
       api_key=getpass.getpass("Please enter your WML api key (hit enter): ")
   )
   ```

2. **Project ID**:
   The Foundation Model requires a project ID for context. You can set it using environment variables or provide it directly:

   ```python
   import os
   try:
       project_id = os.environ["PROJECT_ID"]
   except KeyError:
       project_id = input("Please enter your project_id (hit enter): ")
   ```

3. **Dataset**:
   We will be using a dataset of customer feedback from electronic devices. The data contains customer comments and satisfaction scores.

   ```python
   import pandas as pd
   url = "https://raw.githubusercontent.com/rahul-bhave/watsonx_ai_maverics/main/electronic_device_customer_feedback.csv"
   df = pd.read_csv(url)
   data = df[['Customer_Service', 'Satisfaction']]
   ```

4. **Train/Test Split**:
   The data is split into training and testing sets:

   ```python
   from sklearn.model_selection import train_test_split
   train, test = train_test_split(data, test_size=0.5)
   comments = list(test.Customer_Service)
   satisfaction = list(test.Satisfaction)
   ```

---

## Model Overview

We use IBM's **Granite-13b-Instruct-V2** model for text generation and sentiment analysis. This model is part of the IBM Foundation Models and is fine-tuned for multiple tasks such as question answering, summarization, and classification.

To view all available models:

```python
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
print([model.name for model in ModelTypes])
```

We initialize the model using the `granite-13b-instruct-v2` variant:

```python
from ibm_watsonx_ai.foundation_models import ModelInference
model = ModelInference(
    model_id=ModelTypes.GRANITE_13B_INSTRUCT_V2, 
    params={
        GenParams.MIN_NEW_TOKENS: 0,
        GenParams.MAX_NEW_TOKENS: 1,
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.REPETITION_PENALTY: 1
    },
    credentials=credentials,
    project_id=project_id
)
```

---

## Data

The dataset contains customer feedback comments and their satisfaction labels (`1` for satisfied, `0` for unsatisfied).

Example data:

| Customer_Service                                 | Satisfaction |
|--------------------------------------------------|--------------|
| The device is good.                              | 1            |
| The screen resolution on this laptop is amazing.
|  but it tends to heat up quickly.                | 0            |
| Fantastic device,really happy with this purchase!| 1            |

---

## Model Inference and Evaluation

The model is given a prompt to determine if a customer is satisfied based on their comment:

```python
instruction = """Determine if the customer was satisfied with the experience based on the comment. Return simple yes or no."""
prompt1 = "\n".join([instruction, "Comment:" + comments[2], "Satisfied:"])
result = model.generate_text(prompt=prompt1)
print(result)  # Output: no/yes
```

We analyze a sample of customer feedback using the model and compare the results with actual satisfaction labels:

```python
from sklearn.metrics import accuracy_score
sample_size = 10
prompts_batch = ["\n".join([instruction, "Comment:" + comment, "Satisfied:"]) for comment in comments[:sample_size]]
results = model.generate_text(prompt=prompts_batch)

label_map = {0: "no", 1: "yes"}
y_true = [label_map[sat] for sat in satisfaction][:sample_size]

accuracy = accuracy_score(y_true, results)
print('Accuracy:', accuracy)
```

---

## Results

In this test, the model achieved an accuracy of `90%` when analyzing 10 customer feedback comments.

- **True Labels**: `['no', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes']`
- **Predicted Labels**: `['no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']`

---

## Productivity Improvement

Implementing Watsonx AI models like **Granite-13b-Instruct-V2** significantly enhances productivity when analyzing customer feedback. Traditionally, sentiment analysis requires manual review or complex rule-based systems. With this AI-driven approach, businesses can now automate the sentiment classification process, making it both scalable and accurate. 

By reducing the manual effort in analyzing customer satisfaction data, companies can refocus resources on improving customer service and product development. Additionally, with the automation of sentiment analysis, businesses can rapidly adapt to customer feedback trends, ensuring timely and data-driven decisions that enhance overall customer experience. This results in a more agile and responsive business process.

---

## Conclusion

This project demonstrates the use of IBM's Granite-13b-Instruct-V2 Foundation Model for customer satisfaction analysis. The model was able to accurately predict customer sentiment based on feedback, achieving a high accuracy score on a sample test set.

With further refinement, this system can help businesses understand customer experiences and improve their products.

---

### License
This project is licensed under the MIT License - see the LICENSE file for details.



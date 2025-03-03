# Hybrid_Learning_Model_Fraud_Detection


📌 Project Overview
This repository contains hybrid machine learning models designed to detect illicit cryptocurrency transactions using the Elliptic Bitcoin dataset. Our approach enhances traditional models by integrating deep learning (MLP, DNN) with ensemble methods (Random Forest, Decision Trees) and graph-based neural networks (GCN, GAT). These models improve anomaly detection in blockchain transactions, particularly for AML/CFT (Anti-Money Laundering and Countering the Financing of Terrorism) compliance.

Our research builds upon the base paper:
📄 Detecting Anomalous Cryptocurrency Transactions: An AML/CFT Application of Machine Learning-Based Forensics (Electronic Markets, 2023)

The base paper found that Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) outperform traditional machine learning classifiers, but we extend this work by introducing hybrid models that significantly improve classification accuracy.

📂 Dataset: Elliptic Bitcoin Dataset
The Elliptic dataset consists of 203,769 Bitcoin transactions structured as a graph network. Each transaction is labeled as:

Licit (42,019) – Legitimate transactions
Illicit (4,545) – Fraudulent transactions
Unknown (157,205) – Transactions with no labels
Feature Types:

94 transaction features (TX) – Raw blockchain attributes
73 aggregated features (TX+AGG) – Statistical properties derived from transaction graphs
Our models are trained using the TX and TX+AGG features to identify illicit transactions with greater precision.

🛠 Implemented Models
This repository includes multiple machine learning and deep learning models, each designed to improve fraud detection in cryptocurrency transactions.

1️⃣ MLP + Random Forest (Hybrid Model)
🔹 Description: This model uses a Multi-Layer Perceptron (MLP) for deep feature extraction, followed by a Random Forest classifier for final classification.
🔹 Why this approach?

MLP learns hidden patterns in the transaction data.
Random Forest improves interpretability and reduces overfitting.
🔹 Implementation Steps:
Train MLP on transaction features.
Extract deep embeddings from the hidden layers.
Train a Random Forest classifier on these embeddings.
2️⃣ Decision Tree + MLP (Hybrid Model)
🔹 Description: A Decision Tree (DT) model is used first to classify transactions, and the results are embedded into an MLP for final classification.
🔹 Why this approach?

Decision Trees are fast and interpretable but can be weak in generalization.
MLP refines the classification, boosting accuracy.
🔹 Implementation Steps:
Train Decision Tree on raw transaction data.
Convert DT predictions into new features.
Train an MLP on the combined features.
3️⃣ MLP + Logistic Regression (Hybrid Model)
🔹 Description: This hybrid model embeds an MLP’s feature representations into a Logistic Regression model.
🔹 Why this approach?

Logistic Regression is simple and efficient but struggles with complex relationships.
MLP extracts useful features, making Logistic Regression more effective.
🔹 Implementation Steps:
Train MLP on transaction data.
Use MLP’s hidden layer outputs as new features.
Train a Logistic Regression classifier on these embeddings.
4️⃣ Deep Neural Network (DNN)
🔹 Description: A fully connected deep neural network (DNN) optimized for anomaly detection.
🔹 Why this approach?

DNN captures non-linear patterns in illicit transactions.
More hidden layers enhance representation learning.
🔹 Implementation Steps:
Normalize and preprocess transaction data.
Use a deep neural architecture with multiple hidden layers.
Apply dropout regularization to prevent overfitting.
5️⃣ Graph Convolutional Network (GCN)
🔹 Description: A Graph Convolutional Network (GCN) that learns transaction embeddings from the graph structure.
🔹 Why this approach?

GCN captures transaction relationships in a network format.
It outperforms traditional ML models for graph-based data.
🔹 Implementation Steps:
Construct transaction graphs from blockchain data.
Train a GCN with two convolutional layers.
Apply node classification to predict illicit transactions.
6️⃣ Graph Attention Network (GAT)
🔹 Description: A Graph Attention Network (GAT) that assigns different importance (attention) to transactions based on their relationships.
🔹 Why this approach?

GAT learns which transactions matter the most.
It provides better context-aware fraud detection.
🔹 Implementation Steps:
Construct a graph representation of transactions.
Train a GAT model with multi-head attention layers.
Predict illicit transactions based on node embeddings.
📊 Results & Performance Analysis
Model	Precision	Recall	F1-Score	Accuracy
Decision Tree + MLP	0.9165	0.8985	0.8812	0.9806
Logistic Regression + MLP	0.9233	0.8913	0.8614	0.9795
Random Forest + MLP (Tx)	0.9598	0.8816	0.8816	0.9786
Random Forest + MLP (Tx+agg)	0.9388	0.8695	0.8097	0.9763
DNN (Tx)	0.7189	0.6557	0.6557	0.9414
DNN (Tx+agg)	0.7979	0.8207	0.8091	0.9622
GAT	0.8169	0.7724	0.8169	0.9579
GCN	0.8330	0.8710	0.9127	0.9127
📌 Key Takeaways:

Random Forest + MLP and GCN achieve the highest precision and recall.
Graph-based models (GCN, GAT) capture network relationships better.
Hybrid models outperform standalone classifiers in fraud detection.

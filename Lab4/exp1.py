import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt

# Given values from the problem description
total_positive_samples = 1000
total_negative_samples = 1000

# Model predictions
false_negatives = 250  # Predicted negative, actual positive
false_positives = 150  # Predicted positive, actual negative

# Calculate True Positives (TP) and True Negatives (TN)
true_positives = total_positive_samples - false_negatives
true_negatives = total_negative_samples - false_positives

# Print TP, FP, TN, FN values
print(f"True Positives (TP): {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"True Negatives (TN): {true_negatives}")
print(f"False Negatives (FN): {false_negatives}")

# Construct the confusion matrix
# In sklearn's confusion_matrix: [[TN, FP], [FN, TP]] for binary classification
conf_matrix = np.array([
    [true_negatives, false_positives],
    [false_negatives, true_positives]
])

print("\nConfusion Matrix:")
print(conf_matrix)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Simulate actual labels (y_true)
y_true_positive = np.ones(total_positive_samples)
y_true_negative = np.zeros(total_negative_samples)
y_true = np.concatenate([y_true_positive, y_true_negative])

# Simulate probability scores (y_scores)
# Scores for actual positive samples (1000 total)
# TP: 750 samples with scores > 0.75
scores_tp = np.random.uniform(0.75, 1.0, true_positives)
# FN: 250 samples with scores < 0.75
scores_fn = np.random.uniform(0.0, 0.749, false_negatives)

# Scores for actual negative samples (1000 total)
# FP: 150 samples with scores > 0.75
scores_fp = np.random.uniform(0.75, 1.0, false_positives)
# TN: 850 samples with scores < 0.75
scores_tn = np.random.uniform(0.0, 0.749, true_negatives)

# Combine scores for y_true = 1 and y_true = 0
y_scores_positive_actual = np.concatenate([scores_tp, scores_fn])
y_scores_negative_actual = np.concatenate([scores_fp, scores_tn])

# Combine all scores corresponding to y_true
y_scores = np.concatenate([y_scores_positive_actual, y_scores_negative_actual])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
ax.grid(True)
plt.show()

print(f"Area Under the Curve (AUC): {roc_auc:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Simulate actual labels (y_true)
y_true_positive = np.ones(total_positive_samples)
y_true_negative = np.zeros(total_negative_samples)
y_true = np.concatenate([y_true_positive, y_true_negative])

# Simulate probability scores (y_scores)
# Scores for actual positive samples (1000 total)
# TP: 750 samples with scores > 0.75
scores_tp = np.random.uniform(0.75, 1.0, true_positives)
# FN: 250 samples with scores < 0.75
scores_fn = np.random.uniform(0.0, 0.749, false_negatives)

# Scores for actual negative samples (1000 total)
# FP: 150 samples with scores > 0.75
scores_fp = np.random.uniform(0.75, 1.0, false_positives)
# TN: 850 samples with scores < 0.75
scores_tn = np.random.uniform(0.0, 0.749, true_negatives)

# Combine scores for y_true = 1 and y_true = 0
y_scores_positive_actual = np.concatenate([scores_tp, scores_fn])
y_scores_negative_actual = np.concatenate([scores_fp, scores_tn])

# Combine all scores corresponding to y_true
y_scores = np.concatenate([y_scores_positive_actual, y_scores_negative_actual])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')
ax.grid(True)
plt.show()

print(f"Area Under the Curve (AUC): {roc_auc:.4f}")
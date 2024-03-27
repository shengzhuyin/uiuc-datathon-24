import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv('encoding_transformed_data_n.csv')
#OLD f = pd.read_csv('floored.csv')
#OLD rf = pd.read_csv('resolved.csv')
mosf = pd.read_csv("mos_n.csv")

codes = mosf.iloc[:,0].tolist()
"""
for index in np.arange(0, len(codes), 1):
    print(f"mos_{index}=" + codes[index])
"""

# Drop rows with missing target, separate target from predictors
df.dropna(axis=0, subset=['resolved'], inplace=True)


df = df.drop(['retailer_code', 'date_of_call', 'time_of_call', 'timestamp_call_key', 'serial', 'reason', 'mos','delinquency_history_13_march', 'delinquency_history_18_march', 'card_activation_status_13_march', 'card_activation_status_18_march' ,'account_open_date_13_march','account_open_date_18_march', 'account_status_13_march', 'account_status_18_march', 'ebill_enrolled_status_13_march' ,'ebill_enrolled_status_18_march','delinquency_history_13_march','delinquency_history_18_march'], axis=1)

# cut_length = len(df) // 100

# df = df.head(cut_length)




y = df.resolved
X = df.drop(['resolved'], axis=1)

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(len(y_train))
print(len(y_test))

# Convert the dataset into DMatrix which is a high-performance XGBoost data structure
dtrain = xgb.DMatrix(X_train, label=y_train, missing = np.nan)
dtest = xgb.DMatrix(X_test, label=y_test, missing = np.nan)

params = {
    'max_depth': 4,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    #'gpu_id': 1  # Uncomment if you need to specify a particular GPU
}
num_boost_round = 100

# Train the model
bst = xgb.train(params, dtrain, num_boost_round)

# Predict the probabilities of the positive class
y_pred = bst.predict(dtest)


predictions = [1 if y > 0.5 else 0 for y in y_pred]

# Assuming `predictions` contains your model's predictions,
# and `y_test` contains the actual 'resolved' values from your test dataset
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# For a more detailed report including precision, recall, and F1-score
print(classification_report(y_test, predictions))



# table of importance attribute
importance = bst.get_score()
# Convert importance dictionary into a list of tuples and sort them
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

""" ignore, printing table in bit_mos_xx format
# Print sorted importance scores
for feature, score in sorted_importance:
    print(f"{feature}: {score}")
"""

for feature, score in sorted_importance:
    if feature.startswith('bit_mos_'):
        # Extract the index after 'bit_mos_'
        index = int(feature.split('_')[-1])
        # Replace feature name with corresponding code from the 'codes' list
        feature_name_replaced = codes[index] if index < len(codes) else feature
        print(f"{feature_name_replaced}: {score}")
    elif feature.startswith('bit_reason_'):
        indexf = int(feature.split('_')[-1])
        feature_name_replaced_f = codes[indexf] if indexf < len(codes) else feature
        print(f"{feature_name_replaced_f}_reason: {score}")

    else:
        print(f"{feature}: {score}")

"""
# old plotting, bad due to ugly names
xgb.plot_importance(bst)
plt.title("XGBoost Feature Importance")
plt.show()
"""
"""
# PLOTTING ALL ATTRIBUTES
# Create a mapping from original feature names to more meaningful names
feature_name_mapping = {}

for feature, score in sorted_importance:
    if feature.startswith('bit_mos_'):
        index = int(feature.split('_')[-1])
        new_name = codes[index] if index < len(codes) else feature
        feature_name_mapping[feature] = new_name
    elif feature.startswith('bit_reason_'):
        indexf = int(feature.split('_')[-1])
        new_name_f = f"{codes[indexf]}_reason" if indexf < len(codes) else feature
        feature_name_mapping[feature] = new_name_f
    else:
        feature_name_mapping[feature] = feature

# Use the mapping to replace feature names in the sorted_importance list
sorted_importance_renamed = [(feature_name_mapping[feature], score) for feature, score in sorted_importance]

# Extract feature names and their importance scores
features = [item[0] for item in sorted_importance_renamed]
importances = [item[1] for item in sorted_importance_renamed]

# Create the plot
plt.figure(figsize=(10, 8))
bars = plt.barh(features, importances)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('XPBoost Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

# Add numbers next to each bar
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + bar.get_width() * 0.05  # Shift the text to the right a bit
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
             va='center')
plt.show()
"""

# Only plotting top N% attributes
# Calculate the number of features that make up the top 50%
top_50_percent_index = len(sorted_importance) // 2  # Integer division to get the halfway point

# Keep only the top 50%
sorted_importance_top_50 = sorted_importance[:top_50_percent_index]

# Mapping original feature names to more meaningful names based on your previous logic
feature_name_mapping = {}
for feature, score in sorted_importance_top_50:
    if feature.startswith('bit_mos_'):
        index = int(feature.split('_')[-1])
        new_name = codes[index] if index < len(codes) else feature
        feature_name_mapping[feature] = new_name
    elif feature.startswith('bit_reason_'):
        indexf = int(feature.split('_')[-1])
        new_name_f = f"{codes[indexf]}_reason" if indexf < len(codes) else feature
        feature_name_mapping[feature] = new_name_f
    else:
        feature_name_mapping[feature] = feature

# Use the mapping to replace feature names in the sorted_importance list of the top 50%
sorted_importance_renamed_top_50 = [(feature_name_mapping[feature], score) for feature, score in sorted_importance_top_50]

# Extract feature names and their importance scores
features_top_50 = [item[0] for item in sorted_importance_renamed_top_50]
importances_top_50 = [item[1] for item in sorted_importance_renamed_top_50]

# Create the plot for the top 50%
plt.figure(figsize=(10, 8))
bars = plt.barh(features_top_50, importances_top_50)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance (Top 50%)')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

# Add numbers next to each bar for the top 50%
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + bar.get_width() * 0.05  # Shift the text to the right a bit for clarity
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')

plt.show()


"""
# DATA Preprocessing block

# MOS code collector
mosf = pd.read_csv("mos.csv")

codes = mosf.iloc[:,0].tolist()
# Now codes contains all the entries from the first column as strings

# Adjusted function to encode mos_code as a binary string
# Now it also accepts an attribute name
def encode_as_binary_string(row, attribute, codes):
    return ''.join(['1' if code in row[attribute].split() else '0' for code in codes])

# Apply the function for 'mos' and 'reason' attributes
#df['encoded_mos_code'] = df.apply(lambda row: encode_as_binary_string(row, 'mos', codes), axis=1)
#df['encoded_reason'] = df.apply(lambda row: encode_as_binary_string(row, 'reason', codes), axis=1)

# Convert "resolved" to binary
# df['resolved'] = df['resolved'].apply(lambda x: 1 if x == 'resolved' else (0 if x == 'floor' else np.nan))

# Extract integers from delinquency history attributes and save them as new attributes
#df['delinquency_history_13_march_int'] = df['delinquency_history_13_march'].str.extract(r'\[(\d+)\]')[0]
#df['delinquency_history_18_march_int'] = df['delinquency_history_18_march'].str.extract(r'\[(\d+)\]')[0]

# Optionally, save the modified DataFrame to a new CSV file
df.to_csv('encoded_main_data_d.csv', index=False)

# Print out the first few rows to check
print(df.head())

"""
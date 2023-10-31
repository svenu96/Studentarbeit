from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming `filtered_data` is your original DataFrame
mf = pd.DataFrame(df)
columns_to_remove = ['timestamp', 'datetime', 'compulsive', 'user yes/no', 'urge', 'tense', 'ignore']
filtered_data = mf.drop(columns=columns_to_remove)

# Separate features (X) and target variable (y)
X = filtered_data[['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz']]
y = filtered_data['labelled']

# Resample the data
sampling_strategy = {
    0: 2626968,  # Keep the majority class as is
    1: 50000,    # Adjust the number of instances for class 1 as needed
    2: 25000     # Adjust the number of instances for class 2 as needed
}

over_sampling = SMOTE(sampling_strategy=sampling_strategy)
under_sampling = RandomUnderSampler(sampling_strategy=sampling_strategy)

imbalanced_pipeline = Pipeline([('over', over_sampling), ('under', under_sampling)])

# Apply the pipeline to your data
X_resampled, y_resampled = imbalanced_pipeline.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest Model with Class Weighting
class_weights = {0: 1, 1: 2626968 / 50000, 2: 2626968 / 25000}

rf_model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

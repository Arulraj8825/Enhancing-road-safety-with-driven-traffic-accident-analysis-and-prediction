 import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load dataset
try:
    df = pd.read_csv('traffic_accidents.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File 'traffic_accidents.csv' not found.")
    df = None

if df is not None:
    # Step 2: Clean data
    df.dropna(inplace=True)  # Remove missing values

    # Drop irrelevant columns if present
    if 'location' in df.columns:
        df.drop('location', axis=1, inplace=True)

    # Step 3: Feature Engineering
    # Encode day of the week
    if 'Day' in df.columns:
        df = pd.get_dummies(df, columns=['Day'], drop_first=True)

    # Extract hour if 'Time' column exists
    if 'Time' in df.columns:
        df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        df.dropna(subset=['Hour'], inplace=True)

    # Step 4: Prepare features and target
    if 'Severity' not in df.columns:
        print("❌ 'Severity' column not found in dataset.")
    else:
        X = df.drop('Severity', axis=1)
        y = df['Severity']

        # Step 5: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Step 6: Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Step 7: Evaluate
        y_pred = model.predict(X_test)
        print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

        # Step 8: Visualization
        if 'Hour' in df.columns:
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x='Hour', palette='coolwarm')
            plt.title("Traffic Accidents by Hour")
            plt.xlabel("Hour of Day")
            plt.ylabel("Number of Accidents")
            plt.tight_layout()
            plt.show()
else:
    print("⚠️ Skipping analysis due to dataset load error.")

output:
 

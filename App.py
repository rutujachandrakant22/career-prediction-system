import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# 1. Load & Preprocess Data
# -----------------------------
df = pd.read_csv("PS2_Dataset.csv")

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Separate features and target
X = df.drop("Suggested Job Role", axis=1)
y = df["Suggested Job Role"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# OneHot Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
X[X.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(
    X[X.select_dtypes(include=['int64', 'float64']).columns]
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 1b. Exploratory Data Analysis (EDA)
# -----------------------------
st.header("Exploratory Data Analysis")

# Job Role Distribution
st.subheader("Job Role Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="Suggested Job Role", data=df, palette="viridis", ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Correlation Heatmap of Numeric Features
st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=['int64','float64']).corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Class Balance
st.subheader("Class Balance")
class_counts = df["Suggested Job Role"].value_counts()
fig3, ax3 = plt.subplots()
sns.barplot(x=class_counts.index, y=class_counts.values, palette="plasma", ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

# Skill Ratings vs Job Roles
st.subheader("Skill Ratings vs Job Roles")
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.boxplot(x="Suggested Job Role", y="coding skills rating", data=df, palette="Set2", ax=ax4)
plt.xticks(rotation=45)
st.pyplot(fig4)

# -----------------------------
# 2. Train Models (Fine-Tuned)
# -----------------------------
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- Random Forest with class balancing & tuning ---
rf = RandomForestClassifier(
    n_estimators=300,        # more trees
    max_depth=20,            # deeper trees
    min_samples_split=5,     # avoid overfitting
    class_weight="balanced", # handle class imbalance
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# --- Neural Network with better architecture ---
# Compute class weights for imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))

dl = Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    Dense(len(le.classes_), activation="softmax")
])

dl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
           loss="sparse_categorical_crossentropy",
           metrics=["accuracy"])

history = dl.fit(
    X_train, y_train,
    epochs=30,               # more training
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict, # balance classes
    verbose=0
)

dl_loss, dl_acc = dl.evaluate(X_test, y_test, verbose=0)


# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("Career Prediction System")

st.sidebar.header("Input Your Skills")
coding = st.slider("Coding Skills Rating", 0, 10, 5)
public_speaking = st.slider("Public Speaking Points", 0, 10, 5)
hackathons = st.number_input("Hackathons Attended", 0, 50, 0)
introvert = st.selectbox("Introvert", ["Yes", "No"])
certifications = st.selectbox("Certifications", ["Yes", "No"])
workshops = st.selectbox("Workshops", ["Yes", "No"])
subjects = st.selectbox("Interested Subjects", ["AI", "Cloud", "Networking", "Software Development"])

# Convert inputs into dataframe
input_data = pd.DataFrame({
    "coding skills rating": [coding],
    "public speaking points": [public_speaking],
    "hackathons": [hackathons],
    "Introvert": [introvert],
    "certifications": [certifications],
    "workshops": [workshops],
    "Interested subjects": [subjects]
})

# Apply same preprocessing
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Prediction using best model
best_model = rf if rf_acc >= dl_acc else dl
prediction = best_model.predict(input_data)[0]
career = le.inverse_transform([prediction])[0]

st.success(f"Predicted Career: {career}")

# -----------------------------
# 4. Course Recommendation
# -----------------------------
def recommend_course(career):
    courses = {
        "Applications Developer": "Java, C#, Web Development",
        "Database Developer": "SQL, Oracle, Data Warehousing",
        "Mobile Applications Developer": "Android, iOS, Flutter",
        "Software Engineer": "System Design, Algorithms",
        "UX Designer": "UI/UX, Figma, Adobe XD",
        "Web Developer": "HTML, CSS, JavaScript, React",
        "CRM Technical Developer": "Salesforce, Dynamics CRM",
        "Technical Support": "ITIL, Helpdesk Tools",
        "Systems Security Administrator": "Cybersecurity, Firewalls",
        "Software Quality Assurance (QA) / Testing": "Automation Testing, Selenium",
        "Network Security Engineer": "Cybersecurity, Ethical Hacking, Cryptography",
        "Software Developer": "Full Stack Development, Python, Java"
    }
    return courses.get(career, "General Career Development Courses")

st.info(f"Recommended: {recommend_course(career)}")


# -----------------------------
# 5. Model Comparison Plot
# -----------------------------
results = {"Random Forest": rf_acc, "Neural Network": dl_acc}
fig, ax = plt.subplots()
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="magma", ax=ax)
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
st.pyplot(fig)

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].apply(lambda x: iris.target_names[x])

X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

st.title(" Iris Species Classification Dashboard")
st.write("Proyecto final - Data Mining")

st.sidebar.header(" Parámetros de entrada")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
pred = model.predict(user_data)[0]
pred_name = iris.target_names[pred]

st.subheader(" Predicción de especie")
st.success(f"La especie predicha es: **{pred_name.upper()}**")

st.subheader(" Métricas del Modelo")
st.write(f"**Accuracy:** {acc:.3f}")
st.write(f"**Precision:** {precision:.3f}")
st.write(f"**Recall:** {recall:.3f}")
st.write(f"**F1-score:** {f1:.3f}")

st.subheader(" Visualización 3D con el punto ingresado")

fig = px.scatter_3d(
    df,
    x="petal length (cm)",
    y="petal width (cm)",
    z="sepal length (cm)",
    color="species_name",
    title="3D Scatter Plot",
)

fig.add_scatter3d(
    x=[petal_length],
    y=[petal_width],
    z=[sepal_length],
    mode="markers",
    marker=dict(size=8),
    name="Predicción",
)

st.plotly_chart(fig)

st.subheader("📈 Histogramas por característica")
st.bar_chart(df[iris.feature_names])

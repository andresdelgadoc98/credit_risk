# 🧠 Credit Risk Prediction — Machine Learning Model

Proyecto desarrollado como parte de una prueba técnica para una vacante de **Data Scientist**, enfocado en la **predicción de riesgo crediticio** utilizando *machine learning* y buenas prácticas de modelado.

---

## 🎯 Objetivo

Construir un modelo predictivo capaz de estimar la **probabilidad de incumplimiento de pago (default)** a partir de variables financieras y demográficas de los solicitantes.

---

## 🧩 Dataset

Fuente: [Credit Risk Dataset — Kaggle](https://www.kaggle.com/code/adinaabrar/credit-scoring-like-a-pro-ml-model-for-loan-risk)

- 32,581 registros  
- 12 variables (edad, ingresos, tipo de vivienda, intención del préstamo, tasa de interés, etc.)  
- Variable objetivo: `loan_status` (1 = default, 0 = pagado)

---

## ⚙️ Pipeline del modelo

1. **Preprocesamiento**
   - Imputación de valores faltantes  
     - Numéricos → *mediana*  
     - Categóricos → *valor más frecuente*  
   - Codificación categórica con *One-Hot Encoding*  
   - Escalado de variables numéricas con *StandardScaler*  

2. **División de datos**
   - 80 % entrenamiento / 20 % prueba  
   - Estratificación por `loan_status`  

3. **Balanceo de clases**
   - Aplicación de **SMOTE** para compensar el desbalance entre *default* y *no-default*

4. **Modelado**
   - **Logistic Regression** (baseline)  
   - **Random Forest Classifier** (modelo final)  

5. **Evaluación**
   - Métricas: **AUC**, **F1**, **Accuracy**, **Precision**, **Recall**  
   - Curva ROC y Matriz de confusión  
   - Interpretabilidad: *Permutation Importance*  

---

## 📊 Resultados

| Modelo | AUC | F1 | Accuracy |
|---------|------|------|-----------|
| Logistic Regression | 0.871 | 0.648 | 0.815 |
| **Random Forest** | **0.931** | **0.819** | **0.930** |

El modelo **Random Forest** mostró un excelente poder discriminante, con un AUC de **0.93**, reflejando una alta capacidad para diferenciar entre clientes solventes y de alto riesgo.

---

## 🚀 Mejoras futuras

- Ajuste de hiperparámetros con *GridSearchCV* o *Optuna*  
- Calibración de probabilidades (*Platt scaling*, *Isotonic regression*)  
- Implementación de un **dashboard en Power BI** o **Streamlit**  
- Inclusión de nuevas variables de comportamiento y temporalidad  

---

## 🧠 Tecnologías

- Python 3.10  
- Scikit-learn  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- Imbalanced-learn  
- Joblib  

---

## 📬 Autor

**Carlos Andrés Delgado**  
MCC. Ciencia de la Computación  
📧 [andresdelgadoc98@gmail.com](mailto:andresdelgadoc98@gmail.com)  
💻 [LinkedIn](https://www.linkedin.com/in/carlos-andrés-delgado-9788a91a8/)

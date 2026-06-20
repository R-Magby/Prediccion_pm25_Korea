# Prediccion_pm25_Korea
Proyecto de data science usando random forest y técnicas de feature engineering, para la predicción de material particulado de 2.5 mm. 

# Air Korea: Data Science & Machine Learning Pipeline para la Predicción de Material Particulado ($PM_{2.5}$)
Este repositorio contiene el análisis, preprocesamiento y modelado de datos para la clasificación y predicción de los niveles de alerta de contaminación por material particulado fino ($PM_{2.5}$) en Corea del Sur, basándose en la integración de registros históricos de calidad del aire y variables meteorológicas. El enfoque del proyecto combina técnicas avanzadas de Ciencia de Datos con principios de física de la atmósfera.
---
## 1. Enfoque Físico-Meteorológico y Logros del Proyecto
### Enfoque Físico
La dispersión y concentración de contaminantes como el $PM_{2.5}$ son fenómenos físicos altamente influenciados por la dinámica de fluidos atmosféricos y la termodinámica local. Este proyecto integra variables que determinan el estancamiento o la dispersión de partículas, tales como:
*   **Dinámica del Viento:** El arrastre horizontal de partículas finas.
*   **Temperatura y Punto de Rocío (DEW):** Indicadores de estabilidad térmica y potencial condensación de aerosoles secundarios.
*   **Presión Atmosférica:** El gradiente de presión que dicta la subsidencia (estancamiento) o convección (dispersión) del aire.
### Logros y Resultados Clave
*   **Feature Engineering de Lags:** Se descubrió que la calidad del aire tiene "larga memoria" (autocorrelación). La incorporación de lags temporales ($PM_{2.5}$ en $t-1$, $t-3$, etc.) obtuvo la **mayor correlación y poder predictivo** con el valor objetivo.
*   **Desempeño de Modelos:**
    *   **Random Forest v1:** Baseline con Kappa cuadrático de **~0.697** (Seoul) en validación.
    *   **Random Forest v2 (con Feature Engineering):** Optimización mediante búsqueda por grilla (`GridSearchCV` y `TimeSeriesSplit`), alcanzando un Kappa de **~0.725**.
    *   **XGBoost:** Ajustado con parámetros de regularización, logrando un score Kappa de **~0.718**.
---
## 2. Tecnologías y Librerías Utilizadas
El proyecto se ha desarrollado íntegramente en Python utilizando el ecosistema estándar para computación científica y aprendizaje automático:
*   **Procesamiento y Base de Datos:** `pandas`, `numpy`, `sqlite3`
*   **Estadistica y Series Temporales:** `statsmodels.tsa.seasonal` (seasonal decompose), `scipy.fftpack` (análisis espectral de Fourier)
*   **Visualización:** `matplotlib.pyplot`, `seaborn`
*   **Machine Learning:** `scikit-learn` (Random Forest, GridSearchCV, TimeSeriesSplit, preprocessing), `mord` (LogisticAT)
*   **Gradient Boosting:** `xgboost` (XGBClassifier)
---
## 3. Estructura y Secciones del Notebook
### 3.1. Query
*   Carga de datos crudos desde una base de datos SQLite (`database.db`).
*   Cruces e integración (*inner joins*) entre la tabla de calidad de aire (`fact_pollution`) y la tabla de meteorología (`dim_weather`) agregada diariamente por estación (calculando valores máximos y mínimos de temperatura, presión, velocidad del viento, etc.).
### 3.2. EDA (Análisis Exploratorio de Datos)
*   Análisis de tendencias temporales mediante descomposición estacional y cálculo de medias móviles (ventana de 30 días para anular ruido de alta frecuencia).
*   Visualización de correlaciones lineales y no lineales (Pearson & Spearman) mediante mapas de calor. Se constató la multicolinealidad entre temperaturas y presiones, así como la débil correlación lineal individual de los parámetros meteorológicos aislados con el $PM_{2.5}$.
### 3.3. Preproceso (Data Prep)
*   **Imputación de Missings:** Los valores ausentes o erróneos del sensor (codificados físicamente como `999.9` o `9999.9`) fueron imputados utilizando la media de la estación meteorológica correspondiente.
*   **Codificación y Balanceo:** Agrupación de clases (fusión del nivel crítico `Emergencia` con `Preemergencia`) debido a la escasez de muestras en condiciones extremas.
*   **Normalización:** Escalamiento de características numéricas mediante `MinMaxScaler`.
### 3.4. Machine Learning (V1 - Baselines)
Construcción de los primeros modelos de clasificación sin transformaciones complejas. Evaluación inicial del clasificador en validación cruzada:
*   **Random Forest:**
    *   **Seoul:**  Kappa cuadrático de **~0.604**.
    *   **Cheongju:** Kappa cuadrático de **~0.644**
### 3.5. Feature Engineering & Modelos Avanzados (V2)
Aplicación de transformaciones no lineales basadas en la distribución física de las variables:
*   **Transformaciones:** Aplicación de logaritmos a gases altamente asimétricos (`so2`, `pm10`), elevación al cubo para la distancia del viento (`Distance_max`) y al cuadrado la variable `no2`, esto para mejorar la linealidad y disminuir la dispersion.
*   **Random Forest V2:**
    *   **Seoul:** Score de validación mejorado a **~0.664** tras el ajuste de hiperparámetros con división temporal.

*   **XGBoost:**
    *   **Seoul:** Modelo optimizado mediante regularización con un score Kappa de **~0.656**.
*   **LogisticAT:**
    *   **Seoul:** Modelo enfocado en datos ordinales, Kappa de Cohen de **~0.754**
---
## 4. Tareas Pendientes y Metas Futuras
Para evolucionar este proyecto piloto a una solución robusta y de nivel industrial, se han establecido las siguientes metas:
### Metas Técnicas (Modelo Funcional)
*   **Superar el Score Kappa de 0.80:**
    *   Transformar la dirección del viento a componentes cartesianos ($U$ y $V$) para modelar de manera realista la advección física de la contaminación.
    *   Estudiar y aplicar tecnicas de modelos enfocaados en datos ordinales.
    *   Optimizar los umbrales de decisión mediante regresión y algoritmos de optimización de umbrales sobre el set de validación.
*   **Nuevas Fuentes de Datos:** Incorporar densidad poblacional, volumen de tráfico vehicular y días festivos para capturar el factor de emisión de origen antrópico.
### Industrialización (MLOps y Producción)
*   **Modularización del Código:** Migrar las clases y funciones del notebook a módulos de Python limpios y testeables (`src/data/`, `src/features/`, `src/models/`).
*   **Pipeline de Entrenamiento y Predicción:** Diseñar un pipeline automatizado en `scikit-learn` que encadene la imputación, el escalamiento, la ingeniería de variables y la inferencia del modelo en un solo flujo.
*   **Despliegue de API:** Implementar una interfaz de programación de aplicaciones (API REST) usando **FastAPI** o **Flask** para servir predicciones en tiempo real sobre niveles de alerta de calidad de aire en base a pronósticos meteorológicos de 24 horas.

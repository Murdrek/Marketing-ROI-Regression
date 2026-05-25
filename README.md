
# Google Advanced Data Analytics Professional Certificate
## Course 5: Regression Analysis - Marketing Investment & Sales OLS Model

### Project Overview / Descripción del Proyecto

#### English
In this activity, an Ordinary Least Squares (OLS) simple linear regression model is utilized to explore the mathematical relationship between two continuous variables. The workflow encompasses an end-to-end analytical pipeline: constructing and fitting the model, rigorously validating core statistical assumptions, assessing model performance, interpreting regression coefficients, and translating metrics into actionable business insights for stakeholders.

#### Español
En esta actividad, se utiliza un modelo de regresión lineal simple de Mínimos Cuadrados Ordinarios (OLS) para explorar la relación matemática entre dos variables continuas. El flujo de trabajo abarca un pipeline analítico completo: construcción y ajuste del modelo, validación rigurosa de los supuestos estadísticos clave, evaluación del rendimiento del modelo, interpretación de coeficientes y la traducción de métricas en insights de negocio accionables para los stakeholders.

---

### Background & Business Case / Contexto y Caso de Negocio

#### English
As part of an analytics team providing strategic insights on marketing and sales, this project focuses on optimizing promotional budgets to maximize corporate revenue. Using a historical dataset containing campaign expenditures across Television (`TV`), Radio, and Social Media, the analysis isolates and quantifies the direct predictive power of marketing spend over sales outcomes. Senior leadership relies on these findings to make high-stakes decisions regarding future capital allocation, making precision and statistical rigor paramount.

#### Español
Como parte de un equipo de analítica encargado de proveer insights estratégicos de marketing y ventas, este proyecto se enfoca en la optimización de presupuestos promocionales para maximizar los ingresos corporativos. Utilizando un conjunto de datos históricos que contiene gastos de campaña en Televisión (`TV`), Radio y Redes Sociales, el análisis aísla y cuantifica el poder predictivo directo de la inversión publicitaria sobre las ventas. La alta dirección depende de estos hallazgos para la toma de decisiones de alto impacto en la asignación de capital futuro, lo que hace que la precisión y el rigor estadístico sean fundamentales.

---

---

### Project Structure / Estructura del Proyecto

#### English
* **`notebooks/`**: Contains the full Jupyter Notebook (`Activity_Evaluate simple linear regression.ipynb`) featuring end-to-end data processing, exploratory visualizations, and regression diagnostic plots.
* **`code/`**: Contains the standalone production-ready Python script (`Activity_Evaluate simple linear regression.py`) for clean code tracking.
* **`docs/`**: Dedicated directory for high-level documentation, containing the Executive and Technical Reports tailored for stakeholder alignment meetings.

#### Español
* **`notebooks/`**: Contiene el Jupyter Notebook completo (`Activity_Evaluate simple linear regression.ipynb`) con el procesamiento de datos de principio a fin, visualizaciones exploratorias y gráficos de diagnóstico de la regresión.
* **`code/`**: Contiene el script independiente de Python listo para producción (`Activity_Evaluate simple linear regression.py`) para un seguimiento limpio del código.
* **`docs/`**: Directorio dedicado a la documentación de alto nivel, el cual contendrá los Informes Ejecutivos y Técnicos diseñados para reuniones de alineación con los *stakeholders*.

---

## 🚀 PACE Framework Workflow / Flujo de Trabajo PACE

### 1. Plan Stage / Fase de Planificación
* **English:** Defined the core business objective: quantifying the exact return on investment (ROI) of marketing budgets. Standardized the analytical workspace utilizing `pandas`, `matplotlib`, `seaborn`, and `statsmodels`.
* **Español:** Se definió el objetivo de negocio principal: cuantificar el retorno de inversión (ROI) exacto de los presupuestos de marketing. Se estandarizó el entorno de trabajo analítico utilizando `pandas`, `matplotlib`, `seaborn` y `statsmodels`.

### 2. Analyze Stage / Fase de Análisis
* **English:** Conducted Exploratory Data Analysis (EDA) using a scatterplot matrix to map continuous variable relationships. Rigorously verified the 4 foundational assumptions of Ordinary Least Squares (OLS) linear regression:
  1. *Linearity:* Confirmed visually via initial bivariate scatter plots.
  2. *Independence:* Validated through a Durbin-Watson statistic of **2.002**, proving zero residual autocorrelation.
  3. *Normality:* Assessed via histograms and Q-Q plots, showing a balanced error distribution.
  4. *Homoscedasticity:* Confirmed through a uniform residual distribution scatter plot exhibiting constant variance.
     
* **Español:** Se realizó un Análisis Exploratorio de Datos (EDA) utilizando una matriz de dispersión para mapear las relaciones entre variables continuas. Se verificaron rigurosamente los 4 supuestos fundamentales de la regresión lineal por Mínimos Cuadrados Ordinarios (OLS):
  1. *Linealidad:* Confirmada visualmente mediante gráficos de dispersión bivariados iniciales.
  2. *Independencia:* Validada mediante un estadístico de Durbin-Watson de **2.002**, demostrando ausencia de autocorrelación en los residuos.
  3. *Normalidad:* Evaluada a través de histogramas y gráficos Q-Q, mostrando una distribución de errores equilibrada.
  4. *Homocedasticidad:* Confirmada mediante un gráfico de dispersión de residuos uniforme que exhibe varianza constante.

### 3. Construct Stage / Fase de Construcción
* **English:** Built and fitted the mathematical predictive model. The regression line mathematically proves that the model forcedly passes through the dataset coordinate mean $(\bar{X}, \bar{Y}) = (6, 3)$.
  * **Model Equation:** $$\text{Sales} = -0.1263 + 3.5614 \times \text{TV}$$
  * **Model Fit:** Achieved an $R^2$ score of **0.999**, proving that TV spend accounts for 99.9% of sales variance. The coefficient's $p$-value (**0.000**) confirms absolute statistical significance.
* **Español:** Se construyó y ajustó el modelo predictivo matemático. La línea de regresión demuestra matemáticamente que el modelo pasa obligatoriamente por la media ordenada de los datos $(\bar{X}, \bar{Y}) = (6, 3)$.
  * **Ecuación del Modelo:** $$\text{Sales} = -0.1263 + 3.5614 \times \text{TV}$$
  * **Ajuste del Modelo:** Se alcanzó un $R^2$ de **0.999**, demostrando que la inversión en TV explica el 99.9% de la variación en las ventas. El valor $p$ del coeficiente (**0.000**) confirma una significancia estadística absoluta.

### 4. Execute Stage / Fase de Ejecución

#### Strategic Executive Findings / Hallazgos Ejecutivos Estratégicos
* **English:** * **Quantifiable Impact:** Every additional **$1 million** invested in TV advertising yields an average revenue return of **$3.56 million**.
  * **Risk Mitigation:** The model exhibits a tight 95% Confidence Interval ($[3.558, 3.565]$), representing minimal financial volatility for future forecasts.
  * **Strategic Recommendation:** Protect the current TV budget. Since the confidence band maps uncertainty across every value of $X$, further multivariate phases are recommended to evaluate traditional radio and digital channel attribution synergies.
    
* **Español:**
  * **Impacto Cuantificable:** Cada **$1 millón de dólares** adicional invertido en publicidad televisiva genera un retorno promedio de **$3.56 millones de dólares** en ingresos.
  * **Mitigación de Riesgos:** El modelo exhibe un estrecho Intervalo de Confianza del 95% ($[3.558, 3.565]$), lo que representa una volatilidad financiera mínima para proyecciones futuras.
  * **Recomendación Estratégica:** Blindar el presupuesto actual de TV. Dado que la banda de confianza mapea la incertidumbre a lo largo de cada valor de $X$, se recomiendan fases multivariables adicionales para evaluar las sinergias de atribución de los canales de radio tradicional y digital.

---

### Key Takeaways / Aprendizajes Clave
* **English:** Mastered the transformation of statistical metrics (residuals, coefficients, and confidence bands) into high-level business arguments regarding risk and attribution.
* **Español:** Dominio en la transformación de métricas estadísticas (residuos, coeficientes y bandas de confianza) en argumentos de negocio de alto nivel enfocados en riesgo y atribución presupuestaria.

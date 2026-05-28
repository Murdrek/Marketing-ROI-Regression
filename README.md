# Google Advanced Data Analytics Professional Certificate
## Course 5: Regression Analysis - Marketing Investment & Sales OLS Pipeline

---

## Project Overview / Descripción del Proyecto

### English
This repository documents an evolutionary data analytics pipeline exploring Ordinary Least Squares (OLS) linear regression to optimize marketing budgets and predict corporate revenue. The project scales systematically through two phases: **Phase 1** establishes a baseline using simple linear regression (TV spend). **Phase 2** expands into multiple linear regression, documenting a real-world troubleshooting workflow—navigating a severe multicollinearity bottleneck (Radio vs. Social Media) and implementing a successful tactical redesign using categorical encoding (`C(TV)` + `Radio`).

### Español
Este repositorio documenta un pipeline evolutivo de análisis de datos que explora la regresión lineal por Mínimos Cuadrados Ordinarios (OLS) para optimizar presupuestos de marketing y predecir ingresos corporativos. El proyecto escala sistemáticamente a través de dos fases: la **Fase 1** establece una línea base mediante regresión lineal simple (inversión en TV). La **Fase 2** se expande hacia la regresión lineal múltiple, registrando un flujo real de resolución de problemas—navegando un cuello de botella por severa multicolinealidad (Radio vs. Redes Sociales) e implementando un rediseño táctico exitoso mediante codificación categórica (`C(TV)` + `Radio`).

---

## Background & Business Case / Contexto y Caso de Negocio

English
As part of an analytics team providing strategic insights on marketing and sales, this project focuses on optimizing promotional budgets to maximize corporate revenue. Senior leadership relies on these findings to make high-stakes decisions regarding future capital allocation. While traditional isolated channel analyses (Phase 1) offer clarity, real-world cross-channel interaction requires a multivariate approach (Phase 2). Ensuring statistical rigor by auditing assumptions (Linearity, VIF, Homoscedasticity, and Normality) is paramount to mitigate forecasting risks.

Español
Como parte de un equipo de analítica encargado de proveer insights estratégicos de marketing y ventas, este proyecto se enfoca en la optimización de presupuestos promocionales para maximizar los ingresos corporativos. La alta dirección depende de estos hallazgos para la toma de decisiones de alto impacto en la asignación de capital futuro, lo que hace que la precisión y el rigor estadístico sean fundamentales. Mientras que los análisis de canales aislados (Fase 1) ofrecen claridad, la interacción de canales en el mundo real requiere un enfoque multivariado (Fase 2). Garantizar el rigor estadístico mediante la auditoría de supuestos (Linealidad, VIF, Homocedasticidad y Normalidad) es fundamental para mitigar los riesgos de proyección.

---

## Project Structure / Estructura del Proyecto

### English
* **`data/`**: Source environment containing the raw structured data.
  * [marketing_sales_data.csv](./data/marketing_sales_data.csv) — Baseline historical dataset featuring campaign expenditures (TV, Radio, Social Media) and revenue outcomes.
* **`notebooks/`**: Interactive Jupyter Notebooks capturing the analytical evolution.
  * [Activity_Evaluate simple linear regression.ipynb](./notebooks/Activity_Evaluate%20simple%20linear%20regression.ipynb) — Baseline single-channel model.
  * [Activity_Perform multiple linear regression.ipynb](./notebooks/Activity_Perform%20multiple%20linear%20regression.ipynb) — Advanced multi-channel optimization, VIF troubleshooting, and categorical variable modeling.
* **`code/`**: Standalone production-ready Python scripts for clean code tracking.
  * [Activity_Perform multiple linear regression.py](./code/Activity_Perform%20multiple%20linear%20regression.py) — Refactored production script.
* **`docs/`**: Dedicated directory for high-level documentation and executive summaries.

### Español
* **`data/`**: Entorno de origen que contiene los datos estructurados en bruto.
  * [marketing_sales_data.csv](./data/marketing_sales_data.csv) — Conjunto de datos históricos que contiene los gastos de campaña (TV, Radio, Redes Sociales) y los resultados de ingresos.
* **`notebooks/`**: Jupyter Notebooks interactivos que capturan la evolución analítica.
  * [Activity_Evaluate simple linear regression.ipynb](./notebooks/Activity_Evaluate%20simple%20linear%20regression.ipynb) — Modelo inicial de canal único.
  * [Activity_Perform multiple linear regression.ipynb](./notebooks/Activity_Perform%20multiple%20linear%20regression.ipynb) — Optimización multicanal avanzada, evaluación de problemas de VIF y modelado de variables categóricas.
* **`code/`**: Scripts independientes de Python listos para producción para un seguimiento limpio del código.
  * [Activity_Perform multiple linear regression.py](./code/Activity_Perform%20multiple%20linear%20regression.py) — Script de producción refactorizado.
* **`docs/`**: Directorio dedicado a la documentación de alto nivel e informes ejecutivos.
---

## 🚀 PACE Framework Workflow / Flujo de Trabajo PACE

### 1. Plan Stage / Fase de Planificación

* **English:** Defined the core business objective: quantifying the exact return on investment (ROI) and synergies of marketing channels. Standardized the environment utilizing `pandas`, `patsy`, `matplotlib`, `seaborn`, and `statsmodels`.
* **Español:** Se definió el objetivo de negocio principal: cuantificar el retorno de inversión (ROI) exacto y las sinergias de los canales de marketing. Se estandarizó el entorno utilizando `pandas`, `patsy`, `matplotlib`, `seaborn` y `statsmodels`.

### 2. Analyze Stage / Fase de Análisis

* **English:** Conducted Exploratory Data Analysis (EDA) mapping continuous and categorical variable distributions. Rigorously verified the foundational assumptions of OLS across both project phases:
  * **Linearity:** Confirmed visually via bivariate scatter plots. Phase 2 validated multivariate linearity as residuals remained uniformly dispersed around the $y=0$ baseline.
  * **Independence:** Phase 1 achieved a Durbin-Watson statistic of `2.002`. Phase 2 tracked structural independence across multi-channel spend.
  * **Normality:** Evaluated via histograms and Q-Q plots. Phase 2 confirmed a robust Gaussian bell shape for error distributions, with empirical quantiles tracking the theoretical line perfectly.
  * **Homoscedasticity:** Validated through residual scatter plots. Phase 2 residuals exhibited a uniform, constant variance cloud, completely free of geometric funnel patterns.
* **Español:** Se realizó un Análisis Exploratorio de Datos (EDA) mapeando las distribuciones de variables cuantitativas y categóricas. Se verificaron los supuestos fundamentales de OLS en ambas fases:
  * **Linealidad:** Confirmada visualmente mediante gráficos bivariados. La Fase 2 validó la linealidad multivariada ya que los residuos se mantuvieron uniformemente dispersos alrededor de la línea base $y=0$.
  * **Independencia:** La Fase 1 alcanzó un estadístico de Durbin-Watson de `2.002`. La Fase 2 rastreó la independencia estructural entre los gastos multicanal.
  * **Normalidad:** Evaluada mediante histogramas y gráficos Q-Q. La Fase 2 confirmó una robusta campana de Gauss en la distribución de errores, con los cuantiles empíricos siguiendo la línea teórica a la perfección.
  * **Homocedasticidad:** Validada mediante gráficos de dispersión de residuos. Los residuos de la Fase 2 exhibieron una nube de varianza uniforme y constante, libre de patrones de embudo.

### 3. Construct Stage / Fase de Construcción

#### Phase 1: Simple Linear Regression Model
* **Model Equation:** $\text{Sales} = -0.1263 + 3.5614 \times \text{TV}$
* **Fit & Significance:** $R^2 = 0.999$; TV spend accounts for 99.9% of sales variance ($p = 0.000$).

#### Phase 2: Multiple Linear Regression & Redesign
* **Iteration A (The Multicollinearity Bottleneck):** Attempted an OLS model using `Radio` and `Social_Media`. Auditing revealed severe multicollinearity with a **Variance Inflation Factor (VIF) of 5.17**. This variance inflation destabilized coefficients and neutralized the significance of digital channels, preventing the model from reaching safe harbor.
* **Iteration B (The Optimized Categorical Model):** Redesigned the pipeline by replacing the collinear feature with a categorical interpretation of TV investment (`C(TV)` featuring *Low*, *Medium*, and *High* tranches) combined with continuous `Radio` spend.
* **Model Fit:** Achieved an **Adjusted $R^2 = 0.904$**, explaining **90.4%** of sales variance with absolute statistical significance ($p = 0.000$) across all features.
* **VIF Reset:** Multicollinearity was successfully controlled below the strict 5.0 threshold (`C(TV)[T.Low]`: 4.06, `C(TV)[T.Medium]`: 2.22, `Radio`: 2.83).

#### Español:
#### Fase 1: Modelo de Regresión Lineal Simple
* **Ecuación del Modelo:** $\text{Sales} = -0.1263 + 3.5614 \times \text{TV}$
* **Ajuste y Significancia:** $R^2 = 0.999$; la inversión en TV explica el 99.9% de la variación en las ventas ($p = 0.000$).

#### Fase 2: Regresión Lineal Múltiple y Rediseño
* **Iteración A (El cuello de botella de la Multicolinealidad):** Se probó un modelo OLS con `Radio` y `Social_Media`. La auditoría reveló una severa multicolinealidad con un **Factor de Inflación de la Varianza (VIF) de 5.17**. Esta inflación desestabilizó los coeficientes y neutralizó la significancia de los canales digitales, impidiendo que el modelo llegara a buen puerto.
* **Iteración B (El Modelo Categorizado Optimizado):** Se rediseñó el pipeline reemplazando la característica colineal por la inversión en TV tratada de forma categórica (`C(TV)` en tramos *Low*, *Medium* y *High*) en combinación con el gasto continuo en `Radio`.
* **Ajuste del Modelo:** Se alcanzó un **$R^2$ Ajustado $= 0.904$**, explicando el **90.4%** de la variabilidad de las ventas con significancia estadística absoluta ($p = 0.000$) en todas las variables.
* **Reseteo del VIF:** La multicolinealidad se controló exitosamente por debajo del umbral estricto de 5.0 (`C(TV)[T.Low]`: 4.06, `C(TV)[T.Medium]`: 2.22, `Radio`: 2.83).

### 4. Execute Stage / Fase de Ejecución

#### Strategic Executive Findings / Hallazgos Ejecutivos Estratégicos

* **English:**
  * **Quantifiable Impact:** In Phase 1, single-variable TV investment proved highly dominant (+3.56M revenue per M spent). In Phase 2's cross-channel deployment, every additional million invested in `Radio` yields an average revenue return of **$2.97 million**.
  * **The Categorical "Elevator" Effect:** Setting **High TV Budget** as the operational baseline, downscaling to a **Medium TV Budget** structurally slices revenue by **$75.31 million**, while dropping to a **Low TV Budget** severe cuts revenue by **$154.30 million** (holding Radio constant).
  * **Strategic Recommendation:** Maintain high-tier TV investments as an operational baseline to secure market scale, while aggressively fueling continuous Radio spend to capture rapid, linear revenue growth. Digital social channels should be temporarily paused or re-evaluated due to budget canibalization.
* **Español:**
  * **Impacto Cuantificable:** En la Fase 1, la inversión aislada en TV demostró un dominio absoluto (+3.56M en ingresos por millón invertido). En el despliegue multicanal de la Fase 2, cada millón adicional invertido en `Radio` genera un retorno promedio de **$2.97 millones de dólares**.
  * **El Efecto "Ascensor" Categórico:** Tomando el **Presupuesto Alto de TV (High)** como línea de base operativa, reducir el presupuesto a **Medium** recorta estructuralmente los ingresos en **$75.31 millones**, mientras que caer a un presupuesto **Low** reduce drásticamente las ventas en **$154.30 millones** (manteniendo la Radio constante).
  * **Recomendación Estratégica:** Blindar los niveles altos de inversión en TV como base operativa para asegurar la escala del mercado, mientras se impulsa el gasto continuo en Radio para capturar un crecimiento lineal rápido. Las redes sociales digitales deben ser pausadas o reevaluadas temporalmente debido a la canibalización presupuestaria.

---

## Key Takeaways / Aprendizajes Clave

* **English:** Developed a mature data analytics workflow capable of transforming statistical bottlenecks (like failing the VIF assumption) into tactical redesign opportunities. Mastered the implementation and interpretation of dummy variables (`C()`) and 4-quadrant visual diagnostics to defend model stability before international stakeholders.
* **Español:** Desarrollo de un flujo de trabajo analítico maduro capaz de transformar cuellos de botella estadísticos (como fallar el supuesto de VIF) en oportunidades de rediseño táctico. Dominio en la implementación e interpretación de variables dummy (`C()`) y matrices de diagnóstico visual de 4 cuadrantes para defender la estabilidad de los modelos ante stakeholders internacionales.

### Key Takeaways / Aprendizajes Clave
* **English:** Mastered the transformation of statistical metrics (residuals, coefficients, and confidence bands) into high-level business arguments regarding risk and attribution.
* **Español:** Dominio en la transformación de métricas estadísticas (residuos, coeficientes y bandas de confianza) en argumentos de negocio de alto nivel enfocados en riesgo y atribución presupuestaria.

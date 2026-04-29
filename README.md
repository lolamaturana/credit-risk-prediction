# Practica 1 - Modelización en Ingeniería de Datos 

## Curso 2025-26 <img src="https://images.griddo.cunef.edu/logo-cunef-universidad-1272515f-17b3-4169-8bf6-ef63bfffe920" width="190" valign="middle"> 

### Descripción del Proyecto
Este proyecto consiste en la construcción de un pipeline end-to-end de Machine Learning diseñado para predecir el riesgo de impago en préstamos bancarios. El objetivo principal es determinar, basándose en datos históricos, si un cliente será capaz de devolver su préstamo o no.

### Estructura del Proyecto

El desarrollo se divide en módulos siguiendo el estándar de ingeniería de software para Ciencia de Datos:

1. **Clase de preprocesamiento:** src/preprocessing/practica1_preprocessing.py

2. **Clase de filtrado:** src/filtering/practica1_filtering.py

3. **Notebook de referencia:** practica1_notebook.ipynb


### Benchmarking y Comparativa
Este pipeline toma como punto de partida el material didáctico proporcionado en clase por el profesor Miguel Martín. Se ha realizado un análisis crítico de la arquitectura base para implementar mejoras sustanciales en la robustez, el tratamiento de datos no lineales y la ingeniería de variables. 

- **Repositorio de referencia:** https://github.com/mmartinb75/modelizacion_datos_2026
- **Análisis comparativo detallado:** He incluido un archivo llamado `Miguel_VS_Lola.txt` donde detallo las diferencias técnicas entre la implementación base realizada en clase y mi propuesta de 8 variables optimizadas.
- **Datos utilizados:** El pipeline utiliza el fichero `data/variables_withExperts.xlsx`.

### Resultados Finales
Tras un proceso iterativo de entrenamiento y ajuste, estos son los resultados comparativos de los mejores modelos frente al Modelo Base (FICO):

| Modelo | Accuracy | Precision | Recall | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (v1)** | 0.6335 | 0.3067 | 0.6615 | **0.3586** | 0.6982 |
| **Red Neuronal (MLP v2)** | 0.6219 | 0.3035 | 0.6890 | 0.3580 | **0.7010** |
| **Modelo Base (FICO)** | 0.7200 | 0.2600 | 0.2400 | 0.3500 | 0.5920 |
| **SVM (SVC v1)** | 0.6113 | 0.2979 | 0.6960 | 0.3402 | 0.6947 |

#### Conclusión del Benchmarking:
* **Mejora en Detección:** El pipeline avanzado logra triplicar la capacidad de detección de impagos (**Recall**) respecto al modelo base, pasando del 24% al **casi 70%**.
* **Modelo Ganador:** Se selecciona el **HistGradientBoosting (v1)** como modelo final por presentar el mejor equilibrio global y el mayor **PR-AUC (0.3586)**, métrica crítica en entornos desbalanceados.

### Reflexión Estratégica de Negocio (El *Trade-Off*):
¿Por qué elegir un modelo con una precisión cercana al 30%? 
Con una precisión del 30%, asumimos que estamos cometiendo un volumen alto de **Falsos Positivos** (denegando préstamos a clientes que sí pagarían, lo que supone un coste de oportunidad). Sin embargo, se trata de una **decisión estratégica de negocio** para maximizar el **Recall al ~70%**, asegurándonos de que el banco detecta y esquiva la inmensa mayoría de los **Falsos Negativos**. En el sector bancario, dar un crédito a quien no lo va a devolver es lo que genera las verdaderas pérdidas millonarias.
  
### BIBLIOGRAFÍA Y RECURSOS:

#### Preprocessing:
- IterativeImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
- SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
- OrdinalEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
- TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- RobustScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
- TextEncoder: https://skrub-data.org/stable/reference/generated/skrub.TextEncoder.html

#### Filtering:
- DropConstantFeatures: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/DropConstantFeatures.html
- DropCorrelatedFeatures: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/DropCorrelatedFeatures.html
- SelectByShuffling: https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/SelectByShuffling.html

#### Modelos:
- RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- HistGradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- Maquinas de Soporte Vectorial (SVM): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

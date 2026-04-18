# Bank Marketing Experiment Tracking - Orquestación con Prefect

## ¿Qué se hizo?

### 1. **Consolidación de librerías en el Notebook**
Se organizaron todas las librerías del notebook `Experiment_Tracking.ipynb` en la primera celda (Celda 1: "0. LIBRERÍAS Y CONFIGURACIÓN"), incluyendo:
- Librerías de data science: `pandas`, `numpy`, `matplotlib`, `seaborn`
- MLflow para tracking
- Sklearn: pipelines, preprocesadores, modelos, métricas
- **imblearn**: `Pipeline` (corrige el error anterior), `SMOTE`, `RandomUnderSampler`
- Optuna para HPO

### 2. **Orquestación con Prefect**
Se creó `bank_marketing_experiment_tracking_flow.py` que implementa el pipeline completo con:

#### **Tasks (Tareas Prefect)**
1. **`load_and_preprocess_data`**: Carga datos desde parquet
2. **`feature_engineering`**: Crea 7 nuevas features
3. **`prepare_data`**: Train/test split y clasificación de features
4. **`create_preprocessing_pipelines`**: Crea pipelines de sklearn para LR y trees
5. **`find_best_threshold_cv`**: Busca threshold óptimo usando CV (sin leakage)
6. **`optimize_logistic_regression`**: HPO con Optuna para LR
7. **`optimize_random_forest`**: HPO con Optuna para RF
8. **`optimize_xgboost`**: HPO con Optuna para XGB
9. **`generate_report`**: Genera reporte final con rankings

#### **Flow Principal**
`bank_marketing_experiment_flow`: Orquesta todas las tareas y centraliza la lógica

#### **MLflow Integration**
- Tracking automático de parámetros y métricas
- 3 experimentos separados (LR, RF, XGB)
- Autolog de sklearn/xgboost
- Nested runs para trials de Optuna

#### **Prefect Artifacts**
- Reporte markdown con resultados
- Tabla resumen con comparación de modelos

---

## Cómo usar

### **Opción 1: Ejecutar desde línea de comandos**

```bash
# Navega a la carpeta del proyecto
cd c:\Users\Zenbook\Documents\Aprendizaje_nube\Proyecto_final_MLops

# Activa el virtual environment
.\.venv\Scripts\Activate.ps1

# Ejecuta el flow (con parámetros por defecto)
python flows/bank_marketing_experiment_tracking_flow.py

# O con parámetros personalizados
python flows/bank_marketing_experiment_tracking_flow.py \
  --data-path "data/processed/dataset.parquet" \
  --n-trials 15 \
  --mlflow-uri "http://127.0.0.1:5001"
```

### **Opción 2: Con Prefect (recomendado para producción)**

```bash
# Instala Prefect si no lo tienes
pip install prefect

# Ejecuta el flow con Prefect
prefect flow run flows.bank_marketing_experiment_tracking_flow:bank_marketing_experiment_flow

# O con parámetros
prefect flow run flows.bank_marketing_experiment_tracking_flow:bank_marketing_experiment_flow \
  -p data_path="data/processed/dataset.parquet" \
  -p n_trials_per_model=15
```

### **Opción 3: Desde el Notebook**

Importa y ejecuta:
```python
from flows.bank_marketing_experiment_tracking_flow import bank_marketing_experiment_flow

results = bank_marketing_experiment_flow(
    data_path="../data/processed/dataset.parquet",
    n_trials_per_model=10
)
```

---

## Ventajas de la Orquestación

| Aspecto | Notebook | Orquestación Prefect |
|---------|----------|----------------------|
| **Modularidad** | Código lineal | Tareas reutilizables |
| **Monitoreo** | Manual | Automático con Prefect UI |
| **Reintento** | Manual | Automático con `retries` |
| **Artefactos** | Generados ad-hoc | Generados automáticamente |
| **Logs** | Mezclados | Separados por tarea |
| **Escalabilidad** | Limitada | Distribuida (con workers) |
| **Scheduling** | No | Sí (cron, intervals) |

---

## Estructura del Proyecto

```
Proyecto_final_MLops/
├── flows/
│   ├── bank_marketing_flow.py (existente)
│   └── bank_marketing_experiment_tracking_flow.py (NUEVO)
├── notebooks/
│   └── 02-Experiment-Tracking/
│       └── Experiment_Tracking.ipynb (actualizado con librerías consolidadas)
├── data/
│   └── processed/
│       └── dataset.parquet
└── models/
    └── (artefactos de modelos)
```

---

## Configuración de MLflow

El flow usa MLflow autom​áticamente. Asegúrate de tener el servidor corriendo:

```bash
# Terminal 1: Inicia MLflow Server
cd c:\Users\Zenbook\Documents\Aprendizaje_nube\Proyecto_final_MLops\notebooks\02-Experiment-Tracking
mlflow server `
  --host 127.0.0.1 `
  --port 5001 `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlruns

# Terminal 2: Ejecuta el flow
cd c:\Users\Zenbook\Documents\Aprendizaje_nube\Proyecto_final_MLops
python flows/bank_marketing_experiment_tracking_flow.py
```

Luego accede a: `http://127.0.0.1:5001`

---

## Parámetros Configurables

| Parámetro | Tipo | Por Defecto | Descripción |
|-----------|------|-------------|-------------|
| `data_path` | str | `../data/processed/dataset.parquet` | Ruta al dataset procesado |
| `n_trials_per_model` | int | 10 | Número de trials de Optuna por modelo |
| `mlflow_uri` | str | `http://127.0.0.1:5001` | URI del servidor MLflow |

---

## Próximos Pasos

1. **Ejecuta el flow** para validar que funciona sin errores
2. **Revisa MLflow** para ver los detalles de cada trial
3. **Ajusta `n_trials_per_model`** según tu disponibilidad de tiempo/recursos
4. **Personaliza tasks** si necesitas agregar más modelos (LightGBM, CatBoost, etc.)
5. **Scheduling**: Usa Prefect Cloud para ejecutar automáticamente en horarios específicos

---

## Troubleshooting

**Error: "No module named imblearn"**
```bash
pip install imbalanced-learn
```

**Error: "MLflow connection refused"**
- Asegúrate que el servidor MLflow está corriendo en la terminal
- Verifica el puerto 5001: `netstat -ano | findstr :5001`

**Error: "RandomUnderSampler not recognized"**
- El notebook ahora importa `Pipeline` de `imblearn.pipeline` (no `sklearn.pipeline`)

---

**📝 Creado**: 14 de abril de 2026  
**📊 Modelos**: Logistic Regression, Random Forest, XGBoost  
**🛠️ Stack**: Prefect, MLflow, Optuna, scikit-learn, imbalanced-learn

import mlflow
from mlflow.tracking import MlflowClient
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:5001")
client = MlflowClient()

# El run_id del mejor trial de RandomForest
BEST_RUN_ID = "2522dcc54b8b4a9eb93d7f247f93393c"
MODEL_NAME = "bank-marketing-predictor"

# Con autolog, sklearn guarda el modelo como "model"
model_uri = f"runs:/{BEST_RUN_ID}/model"

# Registrar el modelo
model_details = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

version = model_details.version
print(f"Modelo registrado: versión {version}")

# Agregar descripción
client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description="RandomForest Pipeline para Bank Marketing. F1=0.3849, AUC=0.7215. Optimizado con Optuna."
)

# Tags
client.set_model_version_tag(MODEL_NAME, version, "model_type", "random_forest")
client.set_model_version_tag(MODEL_NAME, version, "dataset", "bank_marketing")
client.set_model_version_tag(MODEL_NAME, version, "f1", "0.3849")

# Promover a Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)
print(f"Modelo v{version} promovido a Production")

# ---- Exportar modelo como archivo para Docker ----
print("\nExportando modelo para Docker...")
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Cargar el modelo desde MLflow Registry
model_uri = "models:/bank-marketing-predictor/Production"
pipeline = mlflow.sklearn.load_model(model_uri)

# Guardarlo como archivo .pkl
joblib.dump(pipeline, "model.pkl")
print("Modelo guardado como model.pkl")
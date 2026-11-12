import os
from typing import Optional
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# CARGA DEL PAQUETE DE DESPLIEGUE (MODELO + REGLAS DE PREPROCESAMIENTO)
# -------------------------------------------------------------
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'models', 'modelo_scouting_rf.joblib')

if not os.path.exists(RUTA_MODELO):
    raise FileNotFoundError(
        f"No se encontró el modelo en '{RUTA_MODELO}'. Ejecuta primero: python exportar_modelo.py"
    )

paquete = joblib.load(RUTA_MODELO)
modelo = paquete['modelo']
columnas_entrenamiento = paquete['columnas_X']
posiciones_validas = paquete['posiciones_validas']

# -------------------------------------------------------------
# DEFINICIÓN DE LA API
# -------------------------------------------------------------
app = FastAPI(
    title="API de Scouting Deportivo",
    description="Clasifica a un jugador como Élite (Top 25% de valor comercial) o Estándar.",
    version="1.0.0"
)

# CORS: permite que index.html (abierto desde el navegador) consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class JugadorEntrada(BaseModel):
    """Esquema de validación (Pydantic) del JSON de entrada."""
    jugador: Optional[str] = Field(None, description="Nombre del candidato (opcional)")
    posicion: str = Field(..., description="Posición del jugador")
    edad: float = Field(..., ge=18, le=45, description="Edad en años")
    estatura_cm: Optional[float] = Field(None, ge=140, le=230,
                                         description="Estatura en cm (si se omite, se imputa por mediana)")


@app.get("/")
def estado():
    return {"estado": "API en ejecución", "documentacion": "/docs"}


@app.get("/posiciones")
def listar_posiciones():
    """Posiciones que el modelo conoce (para poblar el formulario web)."""
    return {"posiciones": posiciones_validas}


@app.post("/predict")
def predecir(jugador: JugadorEntrada):
    if jugador.posicion not in posiciones_validas:
        raise HTTPException(
            status_code=400,
            detail=f"Posición no válida: '{jugador.posicion}'. Opciones: {posiciones_validas}"
        )

    # 1. Imputación con las MISMAS reglas aprendidas en entrenamiento
    estatura = jugador.estatura_cm
    if estatura is None:
        estatura = paquete['medianas_estatura'].get(jugador.posicion)
        if estatura is None or pd.isna(estatura):
            estatura = paquete['mediana_global_estatura']

    # 2. Construir el vector de entrada con las columnas exactas del entrenamiento
    #    (las dummies se arman a mano: get_dummies sobre una sola fila perdería la posición)
    fila = {col: 0 for col in columnas_entrenamiento}
    fila['edad'] = jugador.edad
    fila['estatura_cm'] = estatura
    columna_posicion = f"posicion_{jugador.posicion}"
    if columna_posicion in fila:
        fila[columna_posicion] = 1
    X_nuevo = pd.DataFrame([fila], columns=columnas_entrenamiento)

    # 3. Inferencia
    prediccion = int(modelo.predict(X_nuevo)[0])
    probabilidad = float(modelo.predict_proba(X_nuevo)[0, 1])

    return {
        "jugador": jugador.jugador or "Sin nombre",
        "posicion": jugador.posicion,
        "edad": jugador.edad,
        "estatura_cm": round(float(estatura), 1),
        "estatura_imputada": jugador.estatura_cm is None,
        "prediccion": "Élite" if prediccion == 1 else "Estándar",
        "probabilidad_elite": round(probabilidad * 100, 2),
        "umbral_elite_eur": paquete['umbral_elite_eur']
    }

if __name__ == "__main__":
    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, reload=True)

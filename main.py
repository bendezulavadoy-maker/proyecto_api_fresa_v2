from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import os
import uuid
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo con optimizaciones de memoria
model = YOLO("best.pt")
model.overrides['verbose'] = False
model.overrides['nms'] = True

class_names = ['floración', 'fruto_verde', 'fruto_blanco', 'casi_madura', 'madura']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_name = f"temp_{uuid.uuid4()}.jpg"
    
    try:
        with open(temp_name, "wb") as f:
            f.write(await file.read())

        # Inferencia con imagen reducida para ahorrar memoria
        results = model(temp_name, imgsz=416, verbose=False)[0]

        counts = {name: 0 for name in class_names}
        for cls_id in results.boxes.cls:
            cls_id = int(cls_id)
            if cls_id < len(class_names):
                counts[class_names[cls_id]] += 1

        # Liberar resultados de memoria
        del results
        gc.collect()

    finally:
        # Siempre eliminar el archivo temporal
        if os.path.exists(temp_name):
            os.remove(temp_name)

    flores = counts['floración']
    verdes = counts['fruto_verde']
    blancos = counts['fruto_blanco']
    casi = counts['casi_madura']
    maduras = counts['madura']

    total_frutos = verdes + blancos + casi + maduras

    if flores + total_frutos == 0:
        etapa = "Desarrollo vegetativo"
    elif flores > total_frutos:
        etapa = "Floración"
    elif flores > 0 and verdes > flores:
        etapa = "Floración - inicio de fructificación"
    elif verdes + blancos > flores and casi + maduras <= flores:
        etapa = "Fructificación"
    elif casi + maduras > verdes + blancos:
        etapa = "Madurez"
    else:
        etapa = "Etapa no definida claramente"

    return {
        "counts": counts,
        "etapa_fenologica": etapa
    }

@app.get("/")
def root():
    return {"message": "API YOLO de fresas funcionando correctamente"}
@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok"}

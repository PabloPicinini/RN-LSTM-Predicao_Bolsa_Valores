from fastapi import FastAPI, Request
from app.routes.predictions import router as predictions_router
from app.utils.prometheus.monitor import log_requests

# Metadados para o Swagger
tags_metadata = [
    {
        "name": "CSV",
        "description": "Endpoints relacionados à Predição Histórica",
    },
    {
        "name": "Diária",
        "description": "Endpoints relacionados à Predição do dia Atual",
    },
    {
        "name": "Buscar Predições",
        "description": "Endpoints relacionados consulta do banco de dados que foi realizados a Predição do dia Atual",
    },
    {
        "name": "Página Inicial",
        "description": "Endpoint para a página inicial da API",
    },
]

# Instância da aplicação FastAPI
app = FastAPI(
    title="API para Predição de valores do BBAS3",
    description="API para retornar predições com base em informações históricas ou predição do dia atual",
    version="0.0.1",
    openapi_tags=tags_metadata,
)

# Middleware para monitoramento de requests
@app.middleware("http")
async def custom_log_requests(request: Request, call_next):
    return await log_requests(request, call_next)

# Endpoint principal
@app.get(
    "/",
    response_model=dict,
    tags=["Página Inicial"],
    summary="Página Inicial",
    description="Endpoint para retornar a página inicial da API.",
)
def root():
    return {"message": "Essa é a página inicial do app"}

# Rotas
app.include_router(predictions_router, prefix="/predictions")

# Execução principal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

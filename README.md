# UFRO Assistant

Asistente conversacional en español, especializado en normativa y reglamentos de la Universidad de La Frontera (UFRO). Utiliza RAG (Retrieval Augmented Generation) sobre documentos oficiales locales y puede consultar distintos proveedores de LLM (ChatGPT vía OpenAI/OpenRouter, DeepSeek o un Mock sin costo) con modo de comparación.

## Características clave

- Ingesta de PDFs/TXT locales y generación de chunks en Parquet.
- Indexado vectorial con FAISS y modelos Sentence-Transformers (por defecto all-MiniLM-L6-v2).
- Recuperación de contexto y respuestas con formato obligatorio de citas y referencias.
- Múltiples proveedores de LLM: ChatGPT, DeepSeek y Mock; cambio en caliente durante la sesión.
- Modo comparación DeepSeek vs ChatGPT con tiempos medidos.
- Opcional: vector store en Qdrant (Docker) y script de upsert.
- Evaluación offline con conjunto de preguntas (gold set) y métricas agregadas.

## Requisitos

- Python 3.11+ (recomendado 3.12)
- Windows, macOS o Linux (instrucciones abajo usan Windows PowerShell)
- Dependencias Python del archivo `requirements.txt`
- Claves API según el proveedor que quieras usar:
	- ChatGPT: `OPENROUTER_API_KEY` o `OPENAI_API_KEY`
	- DeepSeek: `DEEPSEEK_API_KEY`

## Instalación rápida (Windows PowerShell)

```powershell
# 1) Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Copiar archivo de ejemplo de variables de entorno y editarlo
copy .env.example .env
# Luego abre .env y pega tus claves/ajustes
```

## Variables de entorno (.env)

Edita un archivo `.env` en la raíz del proyecto. Campos más comunes:

- Proveedores LLM
	- `OPENROUTER_API_KEY`: clave de OpenRouter (si usas ChatGPT vía OpenRouter)
	- `OPENAI_API_KEY`: clave directa de OpenAI (opcional si usas base_url de OpenRouter)
	- `OPENAI_BASE_URL`: base URL para el cliente OpenAI (por defecto `https://openrouter.ai/api/v1`)
	- `DEEPSEEK_API_KEY`: clave de DeepSeek (obligatoria si usas DeepSeek)
	- `DEEPSEEK_BASE_URL`: endpoint compatible OpenAI de DeepSeek (por defecto `https://api.deepseek.com/v1/chat/completions`)
	- `DEEPSEEK_MODEL`: `deepseek-chat` (por defecto) o `deepseek-reasoner`

- Embeddings / RAG
	- `EMBED_MODEL`: modelo Sentence-Transformers (por defecto `all-MiniLM-L6-v2`)

- Qdrant (opcional)
	- `QDRANT_URL`: URL completa de Qdrant (si existe, tiene prioridad sobre host/port)
	- `QDRANT_HOST`: host (por defecto `localhost`)
	- `QDRANT_PORT`: puerto (por defecto `6333`)
	- `QDRANT_API_KEY`: clave si tu Qdrant la requiere
	- `QDRANT_COLLECTION`: nombre de colección (por defecto `ufro_chunks`)

- Otros
	- `DEFAULT_TEMPERATURE` (DeepSeek): por defecto `0.7`
	- `DEFAULT_MAX_TOKENS` (DeepSeek): por defecto `2000`
	- `REQUEST_TIMEOUT` (DeepSeek): por defecto `60`

Revisa `.env.example` para un punto de partida.

## Preparar los datos (pipeline RAG)

1) Coloca tus documentos en `data/raw/` (PDF, TXT y MD). Ejemplos ya incluidos en la carpeta.

2) Ingesta: extrae texto y genera `data/processed/chunks.parquet`.

```powershell
python -m rag.ingest
```

3) Embeddings + FAISS: crea `data/index.faiss` y `data/processed/chunks_with_embeddings.parquet`.

```powershell
python -m rag.embed
```

4) (Opcional) Usar Qdrant con Docker y subir los chunks

```powershell
# Levantar Qdrant
docker compose up -d qdrant

# Upsert de vectores (usa EMBED_MODEL y variables Qdrant del .env)
python -m rag.qdrant_upsert
```

Nota: La app principal usa por defecto FAISS local. El soporte Qdrant está disponible vía `rag/vector_store_qdrant.py` para integraciones futuras.

## Ejecutar el asistente (modo interactivo)

```powershell
python app.py
```

Al iniciar puedes elegir el proveedor:

- 1 = ChatGPT (OpenRouter/OpenAI)
- 2 = DeepSeek
- 3 = Mock (sin costo)
- 4 = Compare (compara DeepSeek vs ChatGPT para una consulta)

Comandos dentro del chat:

- `exit`/`salir`/`quit`: salir
- `/prov` o `/provider` o `/cambiar`: cambiar de proveedor en caliente
- `/compare <pregunta>` o `compare` o `4`: comparar DeepSeek vs ChatGPT para esa pregunta

Parámetros útiles:

- `--provider ask|chatgpt|deepseek|mock` (por defecto `ask` muestra menú)
- `--k <int>` cantidad de chunks recuperados (por defecto 5)

Ejemplo:

```powershell
python app.py --provider chatgpt --k 5
```

La consola muestra latencias de retrieve/chat, tokens aproximados y costo estimado según el proveedor.

## Evaluación con gold set (batch)

El módulo `eval/quality_evaluator.py` permite ejecutar una evaluación offline sobre preguntas definidas en `eval/gold_set.jsonl`.

```powershell
python app.py --batch --provider deepseek --gold eval/gold_set.jsonl --k 5
```

Salidas en `eval/`:

- `results_{provider}_{timestamp}.csv`: respuestas y referencias por pregunta
- `summary_{provider}_{timestamp}.json`: métricas agregadas (exact_match, citation_coverage, latencias y costo promedio)

Formato de `eval/gold_set.jsonl` (una por línea):

```json
{"question": "¿Cuál es la nota mínima de aprobación?", "answer": ""}
```

Si `answer` está vacío, la métrica `exact_match` solo considerará presencia literal en la respuesta cuando sea provista.

## Estructura del proyecto (breve)

- `app.py`: CLI interactiva del asistente (RAG + proveedor LLM)
- `rag/ingest.py`: ingesta de PDFs/TXT a `chunks.parquet`
- `rag/embed.py`: genera embeddings y `index.faiss`
- `rag/retrieve.py`: búsqueda con FAISS
- `rag/prompts.py`: prompt del sistema y construcción del prompt de usuario con contexto
- `rag/vector_store_qdrant.py`: utilidades para Qdrant (cliente, ensure_collection, retriever)
- `rag/qdrant_upsert.py`: script para upsert de chunks a Qdrant
- `providers/*`: proveedores LLM (ChatGPT, DeepSeek y Mock)
- `eval/*`: evaluación de calidad y gold set
- `docker-compose.yml`: servicio Qdrant

## Problemas comunes y soluciones

- “No se encontró el índice FAISS o los chunks procesados.”
	- Ejecuta: `python -m rag.ingest` y luego `python -m rag.embed`.

- Páginas PDF sin texto extraíble
	- `pypdf` no extrae texto de imágenes escaneadas; considera OCR previo.

- Descarga del modelo de embeddings tarda/da error
	- El primer uso de `sentence-transformers` descarga el modelo; verifica tu conexión.

- DeepSeek 401/429/5xx
	- Revisa `DEEPSEEK_API_KEY`, límites de tasa y `DEEPSEEK_BASE_URL`.

- ChatGPT sin API Key
	- El proveedor regresa un mensaje de “deshabilitado”; configura `OPENROUTER_API_KEY` o `OPENAI_API_KEY`.

- Qdrant no inicia o puerto ocupado
	- Cambia el puerto en `docker-compose.yml` o apaga el proceso que ocupa `6333`.

## Costos y privacidad

- El costo estimado se calcula de forma aproximada según precios definidos en cada proveedor (pueden variar). Verifica tu proveedor para valores oficiales.
- Los documentos locales no se suben automáticamente a servicios externos; solo se envían los fragmentos seleccionados al proveedor cuando haces una consulta.

## Licencia

Este proyecto es para fines académicos y prácticos. Ajusta y reutiliza según tus necesidades internas.



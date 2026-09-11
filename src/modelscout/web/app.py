from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from modelscout.dataset.updater import start_background_model_update
from modelscout.web.api import router as api_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_background_model_update(timeout=8.0)
    yield


app = FastAPI(
    title="ModelScout Web",
    description="Find local AI models that fit your hardware, with memory and speed estimates.",
    version="0.1.0",
    lifespan=lifespan,
)

# This dashboard reads local hardware and can run a benchmark on the host.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "[::1]"],
)


@app.middleware("http")
async def local_browser_requests(request: Request, call_next):
    origin = request.headers.get("origin")
    expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
    if (origin and origin != expected_origin) or request.headers.get("sec-fetch-site") == "cross-site":
        return JSONResponse({"detail": "Open ModelScout from its local address."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    )
    return response


app.include_router(api_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ModelScout Web API is running. See /api/health or /docs."}

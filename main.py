"""
FastAPI application and endpoints
"""
import asyncio
import os
import time
import json
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, ValidationError, HttpUrl
from dotenv import load_dotenv
from typing import Any, Dict

from models import QuizRun
from executor import run_pipeline
from test_quizzes import router as test_router

# Manual override storage (in-memory) - queue of answers to try
# Structure: list of {"answer": answer, "timestamp": time}
manual_override_queue: list = []

# Windows compatibility for asyncio
if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()

app = FastAPI(title="Quiz Solver API")

# Include test quiz router
app.include_router(test_router)

SECRET = os.getenv("SECRET")
if not SECRET:
    raise RuntimeError("Environment variable SECRET is not set.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerifyPayload(BaseModel):
    """Request payload for quiz solving"""
    email: EmailStr
    secret: str
    url: HttpUrl
    
    class Config:
        extra = "allow"


class ManualOverridePayload(BaseModel):
    """Manual answer override - adds answer to queue"""
    secret: str
    answer: Any  # Can be string, number, dict, etc.


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": "Invalid JSON payload",
            "details": exc.errors(),
        },
    )


@app.post("/solve")
async def solve(payload: VerifyPayload):
    """
    Solve quiz endpoint
    
    Receives:
    - email: User email
    - secret: Authentication secret
    - url: Quiz URL to solve
    
    Returns:
    - Status and results of quiz solving
    """
    logger.info(f"[API_RECEIVED] POST /solve")
    logger.info(f"[API_RECEIVED_PAYLOAD] Email: {payload.email}, URL: {str(payload.url)}")
    
    # Validate secret
    if not payload.secret or not isinstance(payload.secret, str):
        logger.error(f"[API_ERROR] Invalid secret format")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if payload.secret != SECRET:
        logger.error(f"[API_ERROR] Invalid secret value")
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    logger.info(f"[API_AUTH] Authentication successful for {payload.email}")
    
    # Start pipeline with timing
    logger.info(f"[PIPELINE_START] Beginning quiz solving pipeline")
    started = time.time()
    pipeline_result = await run_pipeline(payload.email, str(payload.url), manual_override_queue)
    duration = time.time() - started
    logger.info(f"[PIPELINE_END] Pipeline completed in {duration:.2f}s - Success: {pipeline_result.get('success')}")
    
    # Prepare response
    response_data = {
        "status": "ok" if pipeline_result.get("success") else "error",
        "email": payload.email,
        "execution_time": f"{duration:.2f}s",
        "pipeline_result": pipeline_result
    }
    
    logger.info(f"[API_RESPONSE] Status: {response_data['status']}, Duration: {response_data['execution_time']}")
    logger.debug(f"[API_RESPONSE_BODY] {json.dumps(response_data, indent=2)[:500]}...")
    
    # Remove sensitive data from response
    if "pipeline_result" in response_data and "artifacts" in response_data["pipeline_result"]:
        del response_data["pipeline_result"]["artifacts"]
    
    return JSONResponse(status_code=200, content=response_data)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "Quiz Solver API"}


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Quiz Solver API",
        "version": "2.0",
        "endpoints": {
            "POST /solve": "Solve a quiz",
            "POST /override": "Set manual answer override for a quiz URL",
            "DELETE /override": "Clear manual answer override",
            "GET /health": "Health check",
            "GET /test-quiz/{quiz_type}": "Get test quiz (literal/compute/web_api/...)",
            "POST /test-quiz/{quiz_type}/submit": "Submit answer for test quiz",
            "GET /test-data/{filename}": "Get test data files",
            "GET /test-api/config": "Mock API endpoint",
            "GET /test-page/dynamic": "Mock JS-rendered page",
            "GET /": "This endpoint"
        }
    }


@app.post("/override")
async def set_override(payload: ManualOverridePayload):
    """Add a manual answer to the queue - will be tried on current quiz"""
    if payload.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    override_entry = {
        "answer": payload.answer,
        "timestamp": time.time()
    }
    manual_override_queue.append(override_entry)
    logger.info(f"[MANUAL_OVERRIDE] Added answer to queue (position {len(manual_override_queue)}): {payload.answer}")
    
    return {
        "status": "ok",
        "message": f"Answer added to queue (position {len(manual_override_queue)}) - will try on current quiz",
        "answer": payload.answer,
        "queue_position": len(manual_override_queue)
    }


@app.delete("/override")
async def clear_override(secret: str):
    """Clear all manual answer overrides from queue"""
    if secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    count = len(manual_override_queue)
    manual_override_queue.clear()
    logger.info(f"[MANUAL_OVERRIDE] Cleared {count} answers from queue")
    return {"status": "ok", "message": f"Cleared {count} answers from queue"}


@app.get("/override/status")
async def get_override_status():
    """Get status of manual override queue"""
    return {
        "queue_length": len(manual_override_queue),
        "answers": manual_override_queue
    }

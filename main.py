# Windows compatibility for asyncio
import asyncio
if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# main.py
import os, time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, ValidationError, HttpUrl
from dotenv import load_dotenv
from typing import Any, Dict
from playwright.async_api import async_playwright

load_dotenv()  # Load .env if present

app = FastAPI(title="Quiz Solver API")

SECRET = os.getenv("SECRET")

if not SECRET:
    # Fail fast if secret not configured
    raise RuntimeError("Environment variable SECRET is not set.")

async def render_page(url: str) -> Dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=45000)
        content = await page.content()
        text = await page.evaluate("() => document.body.innerText")
        links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        pres = await page.eval_on_selector_all("pre,code", "els => els.map(e => e.innerText)")
        await browser.close()
        return {"html": content, "text": text, "links": links, "code_blocks": pres}

async def run_pipeline(email: str, url: str) -> None:
    # render page
    page = await render_page(url)
    print(page)

class VerifyPayload(BaseModel):
    email: EmailStr
    secret: str
    url: HttpUrl
    # Accept other arbitrary fields without validation errors
    # Store additional fields if needed
    class Config:
        extra = "allow"

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": "Invalid JSON payload",
            "details": exc.errors(),
        },
    )

@app.post("/solve")
async def verify(payload: VerifyPayload):
    # Basic required checks
    if not payload.secret or not isinstance(payload.secret, str):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if payload.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    # start pipeline
    started = time.time()
    await run_pipeline(payload.email, str(payload.url))

    # Success response
    # Echo back minimal info; do not return the secret for security
    response: Dict[str, Any] = {
        "status": "ok",
        "email": payload.email,
        "message": "Secret verified",
    }
    return JSONResponse(status_code=200, content=response)


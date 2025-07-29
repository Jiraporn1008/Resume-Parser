import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from parser import parse_resume

app = FastAPI()

# Enable CORS for all origins (Netlify, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/upload/")
async def handle_upload(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    parsed, raw_text = parse_resume(temp_path)

    os.remove(temp_path)
    print(raw_text)

    return {
        "parsed": parsed.model_dump(),
        "raw": raw_text  # For debugging OCR results
    }

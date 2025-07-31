import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from parser import parse_resume

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = [".pdf", ".docx", ".png", ".jpg", ".jpeg"]

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/upload/")
async def handle_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "success": False,
            "status": "error",
            "message": "Validation error: only pdf, doc, docx, png, jpg are allowed"
        }

    temp_path = f"temp_{file.filename}"

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        parsed, raw_text = parse_resume(temp_path)
        print(raw_text)

        return {
            "success": True,
            "status": "success",
            "message": "Resume processed successfully",
            "data": parsed.model_dump()
        }

    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


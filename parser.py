import os
import re
import signal
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr, constr, confloat

from PIL import Image
import easyocr
import pytesseract
from docx import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader


# ============================
# Pydantic Models
# ============================

class PersonalInformation(BaseModel):
    firstNameEN: str
    lastNameEN: str
    firstNameTH: str
    lastNameTH: str
    birthDate: Optional[str]
    age: Optional[int]
    gender: Optional[str] = Field(default=None, pattern="^(Male|Female|Other)$")
    phone: constr(min_length=10, max_length=12)
    email: EmailStr
    province: Optional[str]
    district: Optional[str]

class Salary(BaseModel):
    lastedSalary: Optional[confloat(ge=0)]
    expectSalary: Optional[confloat(ge=0)]

class Qualification(BaseModel):
    industry: Optional[str]
    experiencesYear: Optional[int]
    majorSkill: Optional[str]
    minorSkill: Optional[str]

class Certificate(BaseModel):
    course: Optional[str]
    year: Optional[str]
    institute: Optional[str]

class Experience(BaseModel):
    company: Optional[str]
    position: Optional[str]
    project: Optional[str]
    startDate: Optional[str]
    endDate: Optional[str]
    responsibility: Optional[str]

class Education(BaseModel):
    degreeLevel: str
    program: str
    major: str
    year: str
    university: str

class Resume(BaseModel):
    personalInformation: Optional[PersonalInformation]
    availability: Optional[str]
    currentPosition: Optional[str]
    salary: Optional[Salary]
    qualification: Optional[List[Qualification]]
    softSkills: Optional[List[str]]
    technicalSkills: Optional[List[str]]
    experiences: Optional[List[Experience]]
    educations: Optional[List[Education]]
    certificates: Optional[List[Certificate]]


# ============================
# Prompt & Model Setup
# ============================

resume_template = """
You are an AI assistant tasked with extracting structured information from a technical resume.
Only extract the information that is present in the Resume class.

Resume Detail:
{resume_text}
"""

parser = PydanticOutputParser(pydantic_object=Resume)
prompt_template = PromptTemplate(template=resume_template, input_variables=['resume_text'])

# Use the smallest sufficient model
model = init_chat_model(model='gpt-4o-mini', model_provider='openai').with_structured_output(Resume)


# ============================
# OCR + Image Utils
# ============================

# Lazy load EasyOCR Reader (global singleton)
EASYOCR_READER = easyocr.Reader(['en', 'th'], gpu=False, model_storage_directory='./models/.EasyOCR', download_enabled=False)

class TimeoutException(Exception): pass

def timeout_handler(signum, frame): raise TimeoutException()

def extract_easyocr_text(file_path: str):
    print(f"[EasyOCR] Running EasyOCR on image: {file_path}")
    results = EASYOCR_READER.readtext(file_path)
    text = " ".join([res[1] for res in results])
    avg_conf = sum([res[2] for res in results]) / len(results) if results else 0
    print(f"[EasyOCR] Extracted {len(results)} elements with avg confidence: {avg_conf:.2f}")
    return text, avg_conf

def extract_tesseract_text(file_path: str, timeout=120):
    print(f"[Tesseract] Running Tesseract on image: {file_path}")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        image = Image.open(file_path)
        custom_config = "--oem 3 --psm 6 -l eng"
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except Exception as e:
        print(f"[Tesseract] Error: {e}")
        return ""
    finally:
        signal.alarm(0)

def smart_resize_image(path: str):
    try:
        img = Image.open(path).convert("RGB")
        width, height = img.size
        max_size = 850

        if height > width and height > max_size:
            new_height = max_size
            scale = new_height / height
            new_width = int(width * scale)
        elif width >= height and width > max_size:
            new_width = max_size
            scale = new_width / width
            new_height = int(height * scale)
        else:
            return  # No resize needed

        img = img.resize((new_width, new_height), Image.LANCZOS)
        img.save(path, optimize=True, quality=80)
        print(f"[Resize] Resized image to {new_width}x{new_height}")
    except Exception as e:
        print(f"[Resize] Error: {e}")

def extract_text_from_image(file_path: str) -> str:
    smart_resize_image(file_path)
    text, conf = extract_easyocr_text(file_path)
    return text


# ============================
# File-Type Text Extraction
# ============================

def extract_text_from_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ============================
# Resume Parsing Entry Point
# ============================

def parse_resume(file_path: str) -> Resume:
    print(f"[Parse] Start parsing: {file_path}")
    resume_text = extract_text(file_path)
    prompt = prompt_template.invoke({"resume_text": resume_text})
    result = model.invoke(prompt)
    print("[Parse] Resume parsing complete.")
    return result, resume_text

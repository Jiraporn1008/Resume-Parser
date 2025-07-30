import os
import re
import signal
import gc
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

class TimeoutException(Exception): pass

def timeout_handler(signum, frame): raise TimeoutException()

def extract_easyocr_text(file_path: str):
    print(f"[EasyOCR] Running EasyOCR on image: {file_path}")
    render = easyocr.Reader(['en', 'th'], gpu=False, model_storage_directory='./models/.EasyOCR', download_enabled=False)
    results = render.readtext(file_path)
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
        custom_config = "--oem 3 --psm 6 -l tha+eng"
        text = pytesseract.image_to_string(image, lang="tha+eng", config=custom_config)
        return text.strip()
    except Exception as e:
        print(f"[Tesseract] Error: {e}")
        return ""
    finally:
        signal.alarm(0)

def smart_resize_image(path: str, max_width: int = 1000, max_height: int = 1000):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        width, height = img.size

        if height > width:
            if height <= max_height:
                print(f"[Resize] No resizing needed (height={height} <= {max_height})")
                return
            new_height = max_height
            scale_factor = new_height / height
            new_width = int(width * scale_factor)
        else:
            if width <= max_width:
                print(f"[Resize] No resizing needed (width={width} <= {max_width})")
                return
            new_width = max_width
            scale_factor = new_width / width
            new_height = int(height * scale_factor)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(path, quality=80, optimize=True)
        print(f"[Resize] Resized to {new_width}x{new_height}")
    except Exception as e:
        print(f"[Resize] Failed to resize image: {e}")


def extract_text_from_image(file_path: str) -> str:
    smart_resize_image(file_path, max_width=1000, max_height=1000)
    print(f"[Image] Running Tesseract OCR pipeline for: {file_path}")
    text = extract_tesseract_text(file_path)

    if re.search(r'[\u0E00-\u0E7F]', text):
        print("[Image] Thai text detected. Releasing Tesseract memory before EasyOCR.")
        del text
        gc.collect()
        smart_resize_image(file_path, max_width=850, max_height=850)
        print(f"[Image] Running EasyOCR on image: {file_path}")
        text, confidence = extract_easyocr_text(file_path)

    return text

def extract_text_from_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_invalid_emails(text: str) -> str:
    pattern = r'[\w\.-]+@[\w\-]+'
    matches = re.findall(pattern, text)
    for match in matches:
        if "@" in match and "." not in match.split("@")[1]:
            print(f"[Warning] Found invalid email format: {match}")
            text = text.replace(match, "")
    return text

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def parse_resume(file_path: str) -> Resume:
    print(f"[Parse] Start parsing: {file_path}")
    resume_text = extract_text(file_path)
    resume_text = clean_invalid_emails(resume_text)
    prompt = prompt_template.invoke({"resume_text": resume_text})
    result = model.invoke(prompt)
    print("[Parse] Resume parsing complete.")
    return result, resume_text

import os
import re
import signal
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr, constr, confloat

import pytesseract
import easyocr
from PIL import Image
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
    phone: constr(min_length=10, max_length=15)
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

prompt_template = PromptTemplate(
    template=resume_template,
    input_variables=['resume_text']
)

model = init_chat_model(model='gpt-4o-mini', model_provider='openai').with_structured_output(Resume)


class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def extract_text_from_pdf(file_path: str) -> str:
    print(f"[PDF] Extracting text from PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    combined_text = "\n".join([doc.page_content for doc in docs])
    print(f"[PDF] Extracted {len(combined_text)} characters from PDF.")
    return combined_text

def extract_text_from_docx(file_path: str) -> str:
    print(f"[DOCX] Extracting text from DOCX: {file_path}")
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    print(f"[DOCX] Extracted {len(text)} characters from DOCX.")
    return text

def extract_text_from_txt(file_path: str) -> str:
    print(f"[TXT] Reading plain text file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        print(f"[TXT] Read {len(text)} characters.")
        return text

def extract_easyocr_text(file_path: str):
    print(f"[EasyOCR] Running EasyOCR on image: {file_path}")
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models/.EasyOCR', download_enabled=False)
    results = reader.readtext(file_path)
    text = " ".join([res[1] for res in results])
    avg_conf = sum([res[2] for res in results]) / len(results) if results else 0
    print(f"[EasyOCR] Extracted {len(results)} elements with average confidence: {avg_conf:.2f}")
    return text, avg_conf

def extract_tesseract_text(file_path: str, timeout=180):
    print(f"[Tesseract] Running Tesseract on image: {file_path}")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        image = Image.open(file_path)
        custom_config = "--oem 3 --psm 6 -l tha+eng"
        text = pytesseract.image_to_string(image, lang="tha+eng", config=custom_config)
        print(f"[Tesseract] Extracted {len(text)} characters.")
        return text.strip()
    except Exception as e:
        print(f"[Tesseract] Error: {e}")
        return ""
    finally:
        signal.alarm(0)

def contains_thai(text: str) -> bool:
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

def count_valid_words(text: str) -> int:
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def smart_resize_image(path: str):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        width, height = img.size

        if height > width:
            # Portrait: resize only if height > 1200
            if height <= 1200:
                print(f"[Resize] No resizing needed (height={height} <= 1200)")
                return
            new_height = 1200
            scale_factor = new_height / height
            new_width = int(width * scale_factor)
        else:
            # Landscape or square: resize only if width > 1000
            if width <= 1200:
                print(f"[Resize] No resizing needed (width={width} <= 1000)")
                return
            new_width = 1200
            scale_factor = new_width / width
            new_height = int(height * scale_factor)

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(path, quality=80, optimize=True)
        print(f"[Resize] Resized to {new_width}x{new_height}")
    except Exception as e:
        print(f"[Resize] Failed to resize image: {e}")


def extract_text_from_image(file_path: str) -> str:
    print(f"[Image] Running resize image for: {file_path}")
    smart_resize_image(file_path, max_width=1200, max_height=1200)
    print(f"[Image] Running Tesseract OCR pipeline for: {file_path}")
    text = extract_tesseract_text(file_path)
    return text


def normalize_thai_phone_number(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("66"):
        digits = "0" + digits[2:]
    if digits.startswith("0") and len(digits) == 10:
        return digits
    return phone


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[-1].lower()
    print(f"[Main] Extracting text from: {file_path} (extension: {ext})")
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

# Load datasets (change paths if needed)
def load_text_set(filepath: str) -> set:
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

valid_provinces = load_text_set("dataset-provinces.txt")
valid_technical_skills = load_text_set("dataset-technical-list.txt")

def validate_province(province: Optional[str], valid_provinces: set) -> str:
    if not province:
        print("[Location] Missing province")
        return ""
    if province not in valid_provinces:
        print(f"[Location] Province not in list: {province}")
        return province
    return province

def validate_technical_skills(skills: Optional[List[str]], valid_skills: set) -> Optional[List[str]]:
    if not skills:
        return skills
    validated = []
    for skill in skills:
        if skill in valid_skills:
            validated.append(skill)
        else:
            print(f"[Technical Skill] Skill not in list: {skill} (keeping original)")
            validated.append(skill)
    return validated

def parse_resume(file_path: str) -> Resume:
    print(f"[Parse] Parsing resume: {file_path}")
    resume_text = extract_text(file_path)
    prompt = prompt_template.invoke({"resume_text": resume_text})
    result = model.invoke(prompt)

    if result.personalInformation:
        if result.personalInformation.phone:
            original = result.personalInformation.phone
            formatted = normalize_thai_phone_number(original)
            print(f"[Phone] Normalized phone number: {original} → {formatted}")
            result.personalInformation.phone = formatted

        # Validate province
        result.personalInformation.province = validate_province(
            result.personalInformation.province, valid_provinces
        )

    # Validate technical skills
    if result.technicalSkills:
        result.technicalSkills = validate_technical_skills(
            result.technicalSkills, valid_technical_skills
        )

    print("[Parse] Resume parsing complete.")
    print("[Parse] Ready to show data.")
    return result, resume_text

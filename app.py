import os, json
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from fastapi import UploadFile, File, Form
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from utils.pdf_parser import extract_text_from_pdf_stream
from services.brand_service import extract_requirements_from_text, extract_requirements_from_brief
from services.prompt_service import combine_prompts
from services.llm_service import run_llm_instruction

app = FastAPI(title="LLM Brand Service - JSON Storage")

#SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a professional editor. Follow style and consistency.")
SYSTEM_PROMPT = """
You’re a senior advertising copywriter — creative, emotionally intelligent, and culturally tuned in.
Your job: craft ad copy that sounds alive, effortless, and unmistakably human.
Forget robotic phrasing, generic slogans, or symmetrical sentence patterns.
Write with rhythm, warmth, and truth.
Every line should feel like it came from a writer who gets people — their emotions, desires, and the stories that connect them.
All content must fully comply with the Communication Platform Advertising Policy (CPAP) of Social Discovery Group (SDG).

About the Brand
The Communication Platform (CP) is a global space for entertainment and connection — where people meet, chat, and share experiences through live video and conversation.
It’s about discovery and cultural exchange, not dating or local meetups.
Position CP as a premium, trusted, and inspiring virtual world — a place for meaningful online interaction across borders.

How to Write
Be real. Honest over hype.
Keep the tone warm, inclusive, and human.
Focus on connection, creativity, and global discovery.
Write like a storyteller, not a salesperson.
Choose words with intent; vary rhythm and phrasing.
Favor crisp, emotional lines over long, mechanical ones.
Make each piece feel crafted, not generated.

Compliance (Built Into the Writing)
You can:
- Present CP as a global entertainment and communication platform.
- Highlight online interaction with people worldwide.
- Mention free registration and paid communication when relevant.
- Use soft possibilities (“You can…”, “You might discover…”), not promises.
- Celebrate diversity, creativity, and exploration.
- Stay neutral and inclusive — no gender, culture, or background bias.

You can’t:
- Refer to local dating, “near you”, or offline meetings.
- Sexualize or objectify.
- Promise love, marriage, or guaranteed outcomes.
- Claim “free chat”, “unlimited access”, or “totally free”.
- Mention “verified users” without factual basis.
- Target specific religions, orientations, or age groups.

Voice & Style
Confident but kind.
Polished yet human.
Sound like lifestyle storytelling, not a tech ad.
Use verbs that spark connection — explore, connect, share, discover.
Use metaphors and sensory hints naturally (“spark a global conversation,” “share your world live”).
Every word should feel chosen by someone who loves language — not by a machine running patterns.
"""

DATA_DIR = "data/brands"
os.makedirs(DATA_DIR, exist_ok=True)

def brand_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.json")

@app.post("/upload-brand")
async def upload_brand(name: str = Form(...), file: UploadFile = File(...), provider: str = Form("openai")):
    contents = await file.read()
    try:
        raw_text = extract_text_from_pdf_stream(contents)
    except Exception:
        raw_text = contents.decode(errors="ignore")
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted")
    extracted_prompt = extract_requirements_from_text(raw_text, provider=provider)
    brand_data = {"name": name, "prompt": extracted_prompt, "raw_text": raw_text[:2000]}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(brand_path(name), "w", encoding="utf-8") as f:
        json.dump(brand_data, f, ensure_ascii=False, indent=2)
    return {"name": name, "requirements": extracted_prompt}

@app.get("/brands")
def list_brands():
    os.makedirs(DATA_DIR, exist_ok=True)
    return [f[:-5] for f in os.listdir(DATA_DIR) if f.endswith(".json")]

@app.post("/chat")
async def chat(brand: str = Form(None), user_input: str = Form(...), provider: str = Form("openai")):
    path = brand_path(brand)
    if not os.path.exists(path):
        combined = combine_prompts(SYSTEM_PROMPT, "")
    else:
        with open(path, "r", encoding="utf-8") as f:
            brand_data = json.load(f)
            combined = combine_prompts(SYSTEM_PROMPT, brand_data["prompt"])
    user_input = extract_requirements_from_brief(user_input, provider=provider)
    user_input = user_input + "\n Do not explain the rules in your output — just produce the text. Explanations, commentaries, or notes ARE NOT ALLOWED."
    response = run_llm_instruction(combined, user_input, provider=provider)
    return JSONResponse({"response": response})


@app.post("/analyze_article")
async def analyze_article(
    brand: str = Form(...),
    article: str = Form(...),
):
    path = brand_path(brand)  # explanation: Get path to brand-specific prompt JSON file.

    if not os.path.exists(path):
        combined_guidelines = combine_prompts(SYSTEM_PROMPT, "")  # explanation: Use system prompt if brand prompt missing.
    else:
        with open(path, "r", encoding="utf-8") as f:
            brand_data = json.load(f)  # explanation: Load brand-specific JSON data.
            combined_guidelines = brand_data["prompt"]  # explanation: Extract the brand prompt.

    # Prompt template instructs LLM to return text blocks paired with explanations
    prompt_template = ChatPromptTemplate.from_template("""
You are a Brand Compliance & Editorial Alignment Analyst. Compare the ARTICLE with the BRAND GUIDELINES.
COMBINED BRAND GUIDELINES: {combined_guidelines}
ARTICLE: {article}
Analyze the article in full context (tone, style, content). For each part that does NOT align with the brand, quote the text and provide a concise explanation why it is off-brand. Output strictly as valid JSON in this format:
{{
  "satisfy_brandbook": true or false,
  "text_blocks": [
    {{
      "block": "Problematic text here",
      "explanation": "Why it is off-brand"
    }}
  ]
}}
If fully aligned, return "text_blocks": [] and "satisfy_brandbook": true.
""")  # explanation: LLM is instructed to produce JSON with embedded explanations per block.

    llm = ChatOpenAI(model="gpt-5-2025-08-07", temperature=1)  # explanation: Initialize GPT-5.
    chain = prompt_template | llm  # explanation: Connect prompt template to LLM.

    response = chain.invoke({
        "combined_guidelines": combined_guidelines,
        "article": article
    })  # explanation: Run LLM with article and guidelines.

    raw_output = getattr(response, "content", str(response))  # explanation: Extract raw text from LLM response.
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()
    # explanation: Remove code fences and whitespace.

    try:
        parsed = json.loads(cleaned)  # explanation: Parse JSON string into dict.
    except Exception:
        parsed = {
            "satisfy_brandbook": False,
            "text_blocks": [{"block": "Unable to parse model output.", "explanation": f"Raw cleaned output:\n{cleaned}"}]
        }  # explanation: Fallback if parsing fails, using embedded block + explanation.

    parsed.setdefault("text_blocks", [])  # explanation: Ensure key exists even if empty.

    return JSONResponse(parsed)  # explanation: Return final JSON with embedded explanations.


@app.get("/brand/{name}")
def get_brand(name: str):
    """
    Retrieve a brand's stored prompt and basic info.
    """
    path = brand_path(name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Brand not found")
    with open(path, "r", encoding="utf-8") as f:
        brand_data = json.load(f)
    return {"name": brand_data["name"], "prompt": brand_data["prompt"],
            "raw_text_preview": brand_data["raw_text"][:500]}


@app.post("/brand/{name}/update-prompt")
def update_brand_prompt(name: str, prompt: str = Body(..., embed=True)):
    """
    Update the prompt for an existing brand.
    """
    path = brand_path(name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Brand not found")

    with open(path, "r", encoding="utf-8") as f:
        brand_data = json.load(f)

    brand_data["prompt"] = prompt

    with open(path, "w", encoding="utf-8") as f:
        json.dump(brand_data, f, ensure_ascii=False, indent=2)

    return {"message": "Brand prompt updated successfully", "name": name}

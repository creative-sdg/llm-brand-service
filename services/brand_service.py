from langchain import LLMChain
from langchain.prompts import PromptTemplate
from .llm_service import get_chat_llm

EXTRACTION_TEMPLATE = """You are an assistant extracting writing and proofreading requirements from a brand guidebook.
Return structured, concise bullet points summarizing tone, voice, style, formatting, and prohibited language.
{guide_text}"""

def extract_requirements_from_text(guide_text: str, provider: str = "openai") -> str:
    llm = get_chat_llm(temperature=1, provider=provider)
    prompt = PromptTemplate(template=EXTRACTION_TEMPLATE, input_variables=["guide_text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"guide_text": guide_text})

EXTRACTION_TEMPLATE_BRIEF = """
You are an expert prompt engineer.
Your task is to read a technical writing brief and turn it into a clear, concise prompt that can be used to instruct another large language model to write the article described in the brief.
Your output should be a final writing prompt, phrased as if you are telling the LLM what to write (not describing the brief).
It should sound like a direct instruction, e.g.:
“Write an informational article about… Include sections on… Use a neutral tone…”
Rules:
Summarize requirements into direct writing instructions.
Use plain, natural language (no JSON or metadata).
Include all essential constraints: goal, audience, tone, structure, SEO focus, and style rules.
Remove unnecessary details (deadlines, word counts, internal notes).
Present the output as a single, self-contained prompt suitable for direct input to a writing model.

Here is the brief:
{guide_text}"""

def extract_requirements_from_brief(guide_text: str, provider: str = "openai") -> str:
    llm = get_chat_llm(temperature=1, provider=provider)
    prompt = PromptTemplate(template=EXTRACTION_TEMPLATE_BRIEF, input_variables=["guide_text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"guide_text": guide_text})
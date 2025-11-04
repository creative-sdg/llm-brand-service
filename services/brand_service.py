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

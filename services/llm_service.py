from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
import os

def get_chat_llm(provider: str = "openai", temperature: float = 0.0, model: str | None = None):
    if provider == "openai":
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-5-2025-08-07")
        return ChatOpenAI(temperature=1, model=model_name)
    raise ValueError(f"Unsupported provider: {provider}")

def run_llm_instruction(system_and_brand_prompt: str, user_input: str, provider: str = "openai") -> str:
    llm = get_chat_llm(provider=provider)
    template = "{system_and_brand}\n\nUser task: {user_input}\n\nPlease respond following the above requirements."
    prompt = PromptTemplate(template=template, input_variables=["system_and_brand", "user_input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"system_and_brand": system_and_brand_prompt, "user_input": user_input})

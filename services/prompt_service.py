def combine_prompts(system_prompt: str, brand_prompt: str) -> str:
    return f"{system_prompt}\n---\nBrand-specific requirements:\n{brand_prompt}"

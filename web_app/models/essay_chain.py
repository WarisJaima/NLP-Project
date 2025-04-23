# essay_chain.py
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def load_lora_hf_pipeline(path, base="t5-base", max_length=540):
    base_model = T5ForConditionalGeneration.from_pretrained(base)
    model = PeftModel.from_pretrained(base_model, path)
    tokenizer = T5Tokenizer.from_pretrained(path)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5,
        no_repeat_ngram_size=4
    )
    return HuggingFacePipeline(pipeline=pipe)

llm_intro = load_lora_hf_pipeline("essay_generation_model/t5_intro_lora_v23", max_length=160)
llm_body1 = load_lora_hf_pipeline("essay_generation_model/t5_body1_lora_v23", max_length=240)
llm_body2 = load_lora_hf_pipeline("essay_generation_model/t5_body2_lora_v23", max_length=288)
llm_concl = load_lora_hf_pipeline("essay_generation_model/t5_conclusion_lora_v23", max_length=96)

prompt_intro = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Write a short and clear INTRODUCTION for this IELTS Writing Task 2 topic:\n\n{topic}\n\n"
        "Requirements:\n"
        "- Paraphrase the topic clearly\n"
        "- State your opinion\n"
        "- Add brief context\n"
        "- Stay strictly on topic. Do NOT introduce unrelated content\n"
        "- Use formal academic tone"
    )
)

prompt_body1 = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Write the FIRST BODY PARAGRAPH for this IELTS Writing Task 2 topic:\n\n{topic}\n\n"
        "Instructions:\n"
        "- Present ONE clear reason to support your opinion\n"
        "- Provide logic and ONE specific example\n"
        "- Avoid vague generalisations and unrelated facts\n"
        "- Keep ideas fully aligned with the topic above"
    )
)

prompt_body2 = PromptTemplate(
    input_variables=["topic", "intro", "body1"],
    template=(
        "Write the SECOND BODY PARAGRAPH for this IELTS Writing Task 2 essay.\n\n"
        "TOPIC: {topic}\n\n"
        "INTRO: {intro}\n\n"
        "BODY 1: {body1}\n\n"
        "Instructions:\n"
        "- Present a CLEAR CONTRASTING viewpoint\n"
        "- Use a new idea and a different example\n"
        "- DO NOT repeat phrases, ideas, or examples from Body 1\n"
        "- Begin with a contrast linker (e.g., 'However', 'On the other hand')\n"
        "- Stay strictly relevant to the given TOPIC"
    )
)

prompt_concl = PromptTemplate(
    input_variables=["topic", "intro"],
    template=(
        "Write a CONCLUSION paragraph for this IELTS Writing Task 2 essay.\n\n"
        "TOPIC: {topic}\n\n"
        "INTRODUCTION: {intro}\n\n"
        "Instructions:\n"
        "- Restate your opinion using DIFFERENT words\n"
        "- Briefly summarise both body paragraphs\n"
        "- End with a strong final sentence (recommendation or reflection)"
    )
)

chain_intro = LLMChain(llm=llm_intro, prompt=prompt_intro, output_key="intro")
chain_body1 = LLMChain(llm=llm_body1, prompt=prompt_body1, output_key="body1")
chain_body2 = LLMChain(llm=llm_body2, prompt=prompt_body2, output_key="body2")
chain_concl = LLMChain(llm=llm_concl, prompt=prompt_concl, output_key="conclusion")

def generate_intro(topic):
    return chain_intro.invoke({"topic": topic})["intro"]

def generate_body1(topic):
    return chain_body1.invoke({"topic": topic})["body1"]

def generate_body2(topic, intro, body1):
    return chain_body2.invoke({"topic": topic, "intro": intro, "body1": body1})["body2"]

def generate_conclusion(topic, intro):
    return chain_concl.invoke({"topic": topic, "intro": intro})["conclusion"]

def stream_essay_sections(topic):
    intro = generate_intro(topic)
    yield "intro", intro
    body1 = generate_body1(topic)
    yield "body1", body1
    body2 = generate_body2(topic, intro, body1)
    yield "body2", body2
    conclusion = generate_conclusion(topic, intro)
    yield "conclusion", conclusion
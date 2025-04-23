from transformers import AutoModelForCausalLM, AutoTokenizer
import os

PROMPT_TEMPLATE = """
In this task, you are required to evaluate an IELTS Writing Task 2 essay. Consider the following four criteria and provide a detailed assessment for each, along with a suggested band score:

## Task Achievement:
- Evaluate how well the candidate has addressed the given task.
- Assess the clarity and coherence of the response in presenting ideas.
- Identify if the candidate has fully covered all parts of the task and supported arguments appropriately.
- Suggested Band Score (Task Achievement): [Insert Score]

## Coherence and Cohesion:
- Assess the overall organization and structure of the essay.
- Evaluate the use of linking devices to connect ideas and paragraphs.
- Identify if there is a logical flow of information.
- Suggested Band Score (Coherence and Cohesion): [Insert Score]

## Lexical Resource (Vocabulary):
- Examine the range and accuracy of vocabulary used in the essay.
- Point out specific mistakes in vocabulary, such as inaccuracies or overuse of certain words and Suggest modified versions or alternatives for the identified mistakes. [list of mistakes and rectify]
- Assess the appropriateness of vocabulary for the given context.
- Suggested Band Score (Lexical Resource): [Insert Score]

## Grammatical Range and Accuracy:
- Evaluate the variety and complexity of sentence structures.
- Point out specific grammatical errors, such as incorrect verb forms or sentence construction and Suggest modified versions or corrections for the identified mistakes. [list of mistakes and rectify]
- Examine the use of punctuation and sentence formation.
- Suggested Band Score (Grammatical Range and Accuracy): [Insert Score]

## Overall Band Score:

- Provide an overall band score for the essay, considering the holistic performance across all criteria.
- Consider the synergy of the essay in meeting the task requirements cohesively.
- Suggested Overall Band Score: [Insert Score]

## Feedback and Additional Comments:
- Provide constructive feedback highlighting specific strengths and areas for improvement.
- Suggest strategies for enhancement in weaker areas.

## Prompt:
{}

## Essay:
{}

## Evaluation:
"""

checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dpo_outputs", "checkpoint-100"))

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
model.eval()

def evaluate_essay(prompt: str, essay: str) -> str:
    print("🧠 Evaluating essay...")
    full_prompt = PROMPT_TEMPLATE.format(prompt.strip(), essay.strip())
    inputs = tokenizer([full_prompt], return_tensors="pt").to(model.device)
    print("🧾 Prompt length:", len(full_prompt))

    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.7,
        repetition_penalty=1.1
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print("✅ Full model output (truncated):", decoded[:200])

    split_marker = "## Evaluation:"
    if split_marker in decoded:
        decoded = decoded.split(split_marker, 1)[-1]
        decoded = split_marker + decoded

    return decoded

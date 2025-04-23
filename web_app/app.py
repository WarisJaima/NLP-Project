# app.py
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from models.essay_chain import generate_intro, generate_body1, generate_body2, generate_conclusion
import json
import time
from models.grammar_correction import correct_text
from models.essay_evaluator import evaluate_essay
from models.essay_scoring import score_essay as model_score_essay
from flask import url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/essay')
def essay_page():
    return render_template("second_page.html")

@app.route('/generate_stream')
def generate_stream():
    topic = request.args.get("topic", "")
    if not topic:
        return "Missing topic", 400

    def stream():
        intro = generate_intro(topic)
        yield f"data: {json.dumps({'section': 'intro', 'text': intro})}\n\n"
        time.sleep(0.5)

        body1 = generate_body1(topic)
        yield f"data: {json.dumps({'section': 'body1', 'text': body1})}\n\n"
        time.sleep(0.5)

        body2 = generate_body2(topic, intro, body1)
        yield f"data: {json.dumps({'section': 'body2', 'text': body2})}\n\n"
        time.sleep(0.5)

        conclusion = generate_conclusion(topic, intro)
        yield f"data: {json.dumps({'section': 'conclusion', 'text': conclusion})}\n\n"

    return Response(stream_with_context(stream()), content_type='text/event-stream')

@app.route('/score', methods=['POST'])
def score_essay():
    return jsonify({"message": "Essay scoring coming soon!"}), 501

@app.route('/correct', methods=['POST'])
def correct_grammar():
    return jsonify({"message": "Grammar correction coming soon!"}), 501

@app.route("/grammar_correct", methods=["POST"])
def grammar_correct():
    data = request.get_json()
    original_text = data.get("text", "")
    corrected_html = correct_text(original_text)
    return jsonify({"corrected": corrected_html})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    essay = data.get("essay", "")
    
    print("📥 Received /evaluate request")
    print("Prompt:", prompt[:100])
    print("Essay:", essay[:100])

    if not prompt or not essay:
        return jsonify({"error": "Missing prompt or essay"}), 400

    feedback = evaluate_essay(prompt, essay)
    return jsonify({"feedback": feedback})

@app.route('/score_essay', methods=['POST'])
def score_essay_route():
    data = request.get_json()
    prompt = data.get("prompt", "")
    essay = data.get("essay", "")

    if not prompt or not essay:
        return jsonify({"error": "Missing prompt or essay"}), 400

    try:
        result = model_score_essay(prompt, essay)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

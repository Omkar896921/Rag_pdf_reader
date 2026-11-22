# server.py
from flask import Flask, render_template, request, jsonify
from rag_engine import build_index, answer_question

app = Flask(__name__)

# Store the index of the currently uploaded PDF in memory.
# This is enough for a beginner project with one user and one PDF.
state = {"name": None, "index": None}


# Route 1: Show the home page
@app.route("/")
def home():
    return render_template("index.html")


# Route 2: Receive the PDF and build the index
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file."}), 400

    try:
        state["index"] = build_index(file.stream)
        state["name"] = file.filename
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "name": file.filename,
        "chunks": len(state["index"]["chunks"]),
    })


# Route 3: Receive a question and return an answer
@app.route("/ask", methods=["POST"])
def ask():
    if state["index"] is None:
        return jsonify({"error": "Please upload a PDF first."}), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "The question cannot be empty."}), 400

    try:
        answer, hits = answer_question(question, state["index"])
    except Exception as e:
        # Return the error as JSON so the browser can show it
        # and the terminal shows the full traceback
        print("ask() failed:", e)
        return jsonify({"error": f"Could not get an answer: {e}"}), 500

    sources = [
        {
            "page": h["page"],
            "score": round(h["score"], 2),
            "text": h["text"][:300],  # short excerpt only
        }
        for h in hits
    ]

    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
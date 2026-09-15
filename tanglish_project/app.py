"""
Flask demo app: a mock comment section that classifies each posted
Tanglish comment in real time using the trained model, and shows a
moderator dashboard of everything that got flagged.

Run:
    python3 app.py
Then open http://localhost:5000 in your browser.
"""

from flask import Flask, jsonify, render_template, request

from predict import classify

app = Flask(__name__)

# In-memory store for the demo. Every comment posted during this run
# gets recorded here so the "moderator dashboard" can show a running
# log. Resets when the server restarts -- that's fine for a hackathon
# demo; a real deployment would use a database instead.
comment_log = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/comment", methods=["POST"])
def post_comment():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty comment"}), 400

    result = classify(text)
    comment_log.append(result)

    return jsonify(result)


@app.route("/api/log")
def get_log():
    return jsonify(list(reversed(comment_log)))


if __name__ == "__main__":
    app.run(debug=True, port=5000)

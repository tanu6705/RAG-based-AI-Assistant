from flask import Flask, render_template, request, jsonify
from rag_engine import get_teaching_assistant_response

app = Flask(__name__)

# Route to show your HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route that the JS "fetch" calls
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_query = data.get("query")
    
    if not user_query:
        return jsonify({"answer": "Please enter a question."}), 400

    # Call the logic function
    ai_response = get_teaching_assistant_response(user_query)
    
    return jsonify({"answer": ai_response})

if __name__ == '__main__':
    # debug=True allows the server to restart automatically when you change code
    app.run(debug=True, port=5000)
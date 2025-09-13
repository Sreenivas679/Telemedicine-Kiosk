from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

# Initialize the global variable lang_code
lang_code = None

@app.route('/')
def home_page():
    return render_template('front_page.html', lang_code=lang_code)

# Flask route for setting the language and then redirecting to state_problem
@app.route('/set-language', methods=['POST'])
def set_language():
    global lang_code  # Access the global lang_code variable
    lang_code = request.form.get('lang_code')
    return redirect(url_for('state_problem'))

# Flask route for the state_problem page
@app.route('/state_problem')
def state_problem():
    global lang_code
    return render_template('state_problem.html', lang_code=f'{lang_code}-IN')

# Create a route to process the transcript from state_problem.html
@app.route('/process_transcript', methods=['POST'])
def process_transcript():
    transcript = request.form.get('transcript')
    print("Received Transcript:", transcript)
    
    # You can send a response back to the client if needed
    response = {'result': transcript}
    return jsonify(response)

# List of questions
questions = [
    "Question 1: Is this the first question?",
    "Question 2: Is this the second question?",
    "Question 3: Is this the third question?",
    "Question 4: Is this the fourth question?",
]

# Initialize variables to track the current question and store answers
current_question = 0
answers = {}

@app.route('/start_questionnaire', methods=['GET', 'POST'])
def start_questionnaire():
    global current_question

    if request.method == 'POST':
        # Handle the POST request from the questionnaire (storing answers)
        answer = request.form.get('answer')
        answers[questions[current_question]] = answer
        current_question += 1

    if current_question < len(questions):
        return render_template('questionnaire.html', current_question=current_question, question=questions[current_question])
    else:
        return redirect(url_for('results'))
    
@app.route('/answer', methods=['POST'])
def answer():
    global current_question
    answer = request.form.get('answer')
    answers[questions[current_question]] = answer
    current_question += 1

    response = {'current_question': current_question, 'total_questions': len(questions)}
    return jsonify(response)

@app.route('/results')
def results():
    print(answers)  # Print the answers dictionary in the backend
    return jsonify(answers)

if __name__ == '__main__':
    app.run(debug=True)

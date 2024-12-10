from flask import Flask, render_template, request, jsonify
import mainyboi  # Replace with the module that contains your RAG chatbot logic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if user_message:
        app.logger.info(f'Received message from user: {user_message}')
    else:
        app.logger.error('No message received from user.')

    # Get the bot response
    try:
        bot_response = mainyboi.get_response(user_message)
        app.logger.info(f'Response from chatbot: {bot_response}')
    except Exception as e:
        app.logger.error(f'Error occurred while getting bot response: {e}')
        bot_response = "Sorry, something went wrong."

    return jsonify({'response': bot_response})


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_message = request.form.get('feedback')
    # Handle the feedback, e.g., save it to a database or log it
    return jsonify({'status': 'Feedback received, thank you!'})

if __name__ == '__main__':
    app.run(debug=True)
# RAG Chatbot with Flask Web Interface

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot using a web interface built with Flask. The chatbot is intended for users to ask specific doubts and details about the policy in question, providing relevant and detailed responses. The RAG model is responsible for generating responses based on user queries, leveraging retrieval techniques to find the most relevant context before generating a response. The chatbot interacts with users through a simple, user-friendly HTML interface.

## Features
- A web interface built with Flask and HTML for easy user interaction.
- Integration with a Retrieval-Augmented Generation (RAG) model for enhanced response generation.
- Context retrieval from a local ChromaDB to support the RAG system.
- Real-time chatbot responses.
- No dependencies like LangChain or any APIs for embedding or inference models were used. We used `nomic-embed-text:latest` for embeddings and `llama 3.2:latest` for inference. Chroma was used for vector database operations.

## Project Structure
```
project-root/
├── flasky                   # Main Flask application file
├── templates/
│   └── index.html           # HTML file for the chat interface
├── chunking                 # Module for chunking input data
├── docu_3                   # Documentation file
├── mainyboi                 # Entry point for the application
├── rag                      # Module for invoking the AI model
├── retreiver                # Module for setting up ChromaDB and retrieving relevant context
├── vector_embedding         # Module for embedding vectors
└── README.md                # Project documentation (this file)
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (optional but recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-chatbot-flask.git
   cd rag-chatbot-flask
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have Flask, flask-cors, and any other necessary packages for your RAG model.

### Configuration
- Update the **Chromadb path** and **collection name** in `retreiver` to point to the correct path where your embeddings are stored.
- Make sure `rag` has the correct configuration to invoke the Ollama model or any other LLM you're using.

### Running the Application
1. Start the Flask server:
   ```bash
   python flasky
   ```
   The server will start on `http://127.0.0.1:5000/` by default.

2. Open your web browser and visit `http://127.0.0.1:5000/` to interact with the chatbot.

### Testing
- You can use Postman or `curl` to test the `/chat` endpoint independently:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"message": "Hello"}' http://127.0.0.1:5000/chat
  ```

## Usage
- Open your browser and navigate to `http://127.0.0.1:5000/`.
- Type a message in the input box and click "Send" to interact with the chatbot.
- The chatbot will provide responses based on the retrieved context and generation model.

## Troubleshooting
1. **No Response or Blank Page**:
   - Check the JavaScript console in your browser for any errors.
   - Ensure that the Flask server is running without errors.

2. **Backend Errors**:
   - Check the Flask logs for any errors related to retrieval or model invocation.
   - Add more detailed logging in `flasky` for better insights.

## Future Improvements
- Add authentication for secure use.
- Improve the UI/UX for a better chatting experience.
- Deploy the chatbot using a production-ready WSGI server (e.g., Gunicorn) and set up proper hosting.

## Contributing
Feel free to submit a pull request if you have any improvements or bug fixes. Issues and feature requests are welcome!

## License
This project is licensed under the MIT License.

## Contact
For questions or suggestions, please reach out to `aathavaleharsh@gmail.com`.

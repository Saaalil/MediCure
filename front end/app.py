from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio  # For async operations
from custom_agent_with_caching import generate_response  # Your async function

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend-backend communication

# Route for the chatbot (asynchronous)
@app.route('/chat', methods=['POST'])
async def chat():
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    try:
        # Await the asynchronous function directly
        response = await generate_response(prompt)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Placeholder for favicon.ico to avoid 404 error
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return no content

if __name__ == '__main__':
    # If you're using Flask >= 2.0, this will automatically handle async routes.
    app.run(host='0.0.0.0', port=5511, debug=True)

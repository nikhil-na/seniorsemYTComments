from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_api import extract_video_id, fetch_and_preprocess_comments
from dotenv import load_dotenv

load_dotenv() 

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hello, Flask!"

# Flask endpoint to download all comments from a given YouTube video URL.
@app.route("/api/fetch_comments", methods=["POST"])
def fetch_comments():
    """
    Flask endpoint to download all comments from a given YouTube video URL.
    Expects a JSON payload with 'video_url'.

    Request Body Example:
    {
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }

    Response Body Example (Success):
    {
        "video_id": "dQw4w9WgXcQ",
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "comment_count": 123,
        "comments": [
            {"authorDisplayName": "User1", "textDisplay": "Great video!", ...},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        video_url = data.get('video_url')

        if not video_url:
            return jsonify({'error': 'video_url is required'}), 400

        video_id = extract_video_id(video_url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400

        # Fetch and preprocess comments
        comments = fetch_and_preprocess_comments(video_id)

        return jsonify({
            "video_id": video_id,
            "video_url": video_url,
            "comment_count": len(comments),
            "comments": comments,
            "stats": {
                "total_fetched": len(comments),
                "total_preprocessed": sum(1 for c in comments if c["cleaned_text"]),
        }
        })
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve comments: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

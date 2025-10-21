from statistics import mean
from typing import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_api import extract_video_id, fetch_and_preprocess_comments, analyze_sentiment
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

# Flask endpoint to perform sentiment analysis on preprocessed comments
@app.route('/api/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    """
    Perform sentiment analysis on all comments for a given YouTube video URL.
    The endpoint fetches comments internally, preprocesses them, and returns sentiment analysis.
    
    Request Body:
        {
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        }
    
    Response:
        {
            "success": true,
            "comments": [...],
            "statistics": {...}
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

        comments = fetch_and_preprocess_comments(video_id)
        if not comments:
            return jsonify({'error': 'No comments found for this video'}), 404
        
        # Analyze sentiment for each comment
        analyzed_comments = []
        sentiment_counts = Counter()
        confidence_counts = Counter()
        compound_scores = []

        for comment in comments:
            cleaned_text = comment.get('cleaned_text') or fetch_and_preprocess_comments(comment.get('text', ''))
            if not cleaned_text:
                continue

            # Proper sentiment analysis function
            sentiment_result = analyze_sentiment(cleaned_text)

            analyzed_comment = {**comment, 'sentiment': sentiment_result}
            analyzed_comments.append(analyzed_comment)

            # Update statistics
            sentiment_counts[sentiment_result['sentiment']] += 1
            confidence_counts[sentiment_result['confidence']] += 1
            compound_scores.append(sentiment_result['compound'])
        
        total = len(analyzed_comments)
        
        # Calculate percentages
        sentiment_distribution = {
            k: round((v / total) * 100, 2) for k, v in sentiment_counts.items()
        }
        
        # Calculate average compound score
        avg_compound = round(mean(compound_scores), 3)
        if avg_compound >= 0.05:
            overall_sentiment = 'positive'
        elif avg_compound <= -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return jsonify({
            'success': True,
            'comments': analyzed_comments,
            'statistics': {
                'total_comments': total,
                'sentiment_counts': dict(sentiment_counts),
                'sentiment_distribution': sentiment_distribution,
                'confidence_counts': dict(confidence_counts),
                'average_compound_score': avg_compound,
                'overall_sentiment': overall_sentiment,
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

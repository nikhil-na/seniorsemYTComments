import googleapiclient.discovery 
import googleapiclient.errors
import os
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Extract video ID from YouTube URL
def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if len(url) == 11 and re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url
    return None

# Get YouTube comments
def get_youtube_comments(video_id):
    """
    Fetches all top-level comments for a given YouTube video ID using the YouTube Data API.
    Handles pagination to retrieve more than 100 comments.

    Args:
        video_id (str): The ID of the YouTube video.
        api_key (str): Your YouTube Data API key.

    Returns:
        list: A list of dictionaries, each representing a comment.
              Returns an empty list if no comments are found or an error occurs.
    """
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
    comments_list = []
    next_page_token = None

    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100, 
                pageToken=next_page_token,
                textFormat="plainText" 
            )
            response = request.execute()

            for item in response['items']:
                comment_data = item['snippet']['topLevelComment']['snippet']
                cleaned_text = preprocess_comments(comment_data['textDisplay'])

                comments_list.append({
                'id': item['snippet']['topLevelComment']['id'],
                'author': comment_data['authorDisplayName'],
                'text': comment_data['textDisplay'],
                'likes': comment_data['likeCount'],
                'published_at': comment_data['publishedAt'],
                'sentiment': analyze_sentiment(cleaned_text)
        })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break 
        except googleapiclient.errors.HttpError as e:
            print(f"An API error occurred while fetching comments for video ID {video_id}: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    return comments_list

# Preprocess comments to remove emojis, extra symbols, and clean text for sentiment analysis
def preprocess_comments(text):
    """
    Preprocess comment text by removing emojis, extra symbols, and cleaning text
    
    Args:
        text (str): Raw comment text
        
    Returns:
        str: Cleaned comment text
    """
    if not text:
        return ""
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra punctuation and symbols (keep basic punctuation for sentiment)
    # Keep: . , ! ? - ' "
    text = re.sub(r'[^\w\s.,!?\-\'\"]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Convert to lowercase for consistency
    text = text.lower()

    # Filter usernames (e.g., @nikhil -> nikhil)
    text = re.sub(r'@(\w+)', r'\1', text)
    
    return text

# Fetch and preprocess comments for sentiment analysis
def fetch_and_preprocess_comments(video_id):
    """Fetch comments and clean them immediately"""
    comments = get_youtube_comments(video_id)

    if not comments:
        return []

    processed_comments = []
    for comment in comments:
        cleaned_text = preprocess_comments(comment['text'])
        processed_comments.append({
            'id': comment.get('id'),
            'author': preprocess_comments(comment.get('author')),
            'text': comment.get('text'),
            'cleaned_text': cleaned_text,
        })

    return processed_comments

# Analyze sentiment using VADER
def analyze_sentiment(text):
    """
    Analyze sentiment using VADER
    VADER is specifically designed for social media text
    
    Args:
        text (str): Cleaned comment text
        
    Returns:
        dict: Sentiment scores and classification
    """
    if not text:
        return {
            'compound': 0.0,
            'positive': 0.0,
            'neutral': 1.0,
            'negative': 0.0,
            'sentiment': 'neutral',
            'confidence': 'low'
        }
    try:
        scores = vader_analyzer.polarity_scores(text)
        
        compound = scores['compound']
        
        # Classify sentiment based on compound score
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        confidence_score = abs(compound)
        
        if confidence_score >= 0.5:
            confidence = 'high'
        elif confidence_score >= 0.2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'compound': round(compound, 3),
            'positive': round(scores['pos'], 3),
            'neutral': round(scores['neu'], 3),
            'negative': round(scores['neg'], 3),
            'sentiment': sentiment,
            'confidence': confidence
        }
        
    except Exception as e:
        return {
            'compound': 0.0,
            'positive': 0.0,
            'neutral': 1.0,
            'negative': 0.0,
            'sentiment': 'neutral',
            'confidence': 'low',
            'error': str(e)
        }
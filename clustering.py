"""
Clustering Functions
Handles K-Means clustering of comments based on sentiment
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def cluster_comments(comments, num_clusters=5):
    """
    Cluster comments based on sentiment scores using K-Means
    
    Args:
        comments (list): List of comments with sentiment data
        num_clusters (int): Number of clusters to create (default: 3)
        
    Returns:
        tuple: (clustered_comments, clusters_info, statistics)
    """
    # Validate input
    if not comments:
        raise ValueError('comments array is required')
    
    if not isinstance(comments, list):
        raise ValueError('comments must be an array')
    
    # Validate that comments have sentiment data
    valid_comments = []
    for comment in comments:
        if 'sentiment' in comment and 'compound' in comment['sentiment']:
            valid_comments.append(comment)
    
    if not valid_comments:
        raise ValueError('comments must have sentiment data')
    
    if len(valid_comments) < num_clusters:
        raise ValueError(f'Need at least {num_clusters} comments for {num_clusters} clusters')
    
    # Extract features for clustering
    features = []
    for comment in valid_comments:
        sentiment = comment['sentiment']
        features.append([
            sentiment['compound'],
            sentiment['positive'],
            sentiment['neutral'],
            sentiment['negative']
        ])
    
    X = np.array(features)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Assign cluster labels to comments
    clustered_comments = []
    cluster_data = {}
    
    for i, comment in enumerate(valid_comments):
        cluster_id = int(cluster_labels[i])
        
        # Initialize cluster data if not exists
        if cluster_id not in cluster_data:
            cluster_data[cluster_id] = {
                'comments': [],
                'compound_scores': [],
                'sentiments': []
            }
        
        # Add comment to cluster
        comment_with_cluster = {
            **comment,
            'cluster': cluster_id
        }
        
        clustered_comments.append(comment_with_cluster)
        cluster_data[cluster_id]['comments'].append(comment_with_cluster)
        cluster_data[cluster_id]['compound_scores'].append(comment['sentiment']['compound'])
        cluster_data[cluster_id]['sentiments'].append(comment['sentiment']['sentiment'])
    
    # Analyze each cluster
    clusters_info = {}
    
    for cluster_id, data in cluster_data.items():
        avg_compound = np.mean(data['compound_scores'])
        
        # Determine cluster label based on average compound score
        if avg_compound >= 0.3:
            label = 'highly_positive'
        elif avg_compound >= 0.05:
            label = 'positive'
        elif avg_compound >= -0.05:
            label = 'neutral'
        elif avg_compound >= -0.3:
            label = 'negative'
        else:
            label = 'highly_negative'
        
        # Count sentiment types in cluster
        sentiment_counts = {
            'positive': data['sentiments'].count('positive'),
            'negative': data['sentiments'].count('negative'),
            'neutral': data['sentiments'].count('neutral')
        }
        
        # Get dominant sentiment
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        clusters_info[str(cluster_id)] = {
            'label': label,
            'dominant_sentiment': dominant_sentiment,
            'count': len(data['comments']),
            'percentage': round((len(data['comments']) / len(valid_comments) * 100), 2),
            'average_compound': round(avg_compound, 3),
            'sentiment_counts': sentiment_counts,
            'comments': data['comments']
        }
        
        # Update comments with cluster label
        for comment in clustered_comments:
            if comment['cluster'] == cluster_id:
                comment['cluster_label'] = label
    
    # Create statistics
    statistics = {
        'total_comments': len(valid_comments),
        'num_clusters': num_clusters,
        'clustering_method': 'K-Means',
        'features_used': ['compound', 'positive', 'neutral', 'negative']
    }
    
    return clustered_comments, clusters_info, statistics
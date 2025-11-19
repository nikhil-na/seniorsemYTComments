const API_BASE_URL = "http://127.0.0.1:5000";

// DOM Elements
const videoUrlInput = document.getElementById("videoUrl");
const numClustersInput = document.getElementById("numClusters");
const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");
const error = document.getElementById("error");
const results = document.getElementById("results");

// Extract video ID from URL
function extractVideoId(url) {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
    /youtube\.com\/shorts\/([^&\n?#]+)/,
  ];

  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }

  if (url.length === 11 && /^[A-Za-z0-9_-]{11}$/.test(url)) {
    return url;
  }

  return null;
}

// Show error message
function showError(message) {
  const errorMessage = document.getElementById("errorMessage");
  errorMessage.textContent = message;
  error.classList.remove("d-none");
  error.classList.add("show");

  // Auto-hide after 5 seconds
  setTimeout(() => {
    const bsAlert = new bootstrap.Alert(error);
    bsAlert.close();
  }, 5000);
}

// Hide error message
function hideError() {
  error.classList.add("d-none");
  error.classList.remove("show");
}

// Analyze comments
async function analyzeComments() {
  const videoUrl = videoUrlInput.value.trim();
  const numClusters = numClustersInput.value || 4;

  if (!videoUrl) {
    showError("Please enter a YouTube URL");
    return;
  }

  const videoId = extractVideoId(videoUrl);
  if (!videoId) {
    showError("Invalid YouTube URL. Please check the URL and try again.");
    return;
  }

  // Show loading, hide results
  loading.classList.remove("d-none");
  results.classList.add("d-none");
  hideError();

  try {
    const response = await fetch(`${API_BASE_URL}/api/cluster-comments`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        video_url: videoUrl,
        num_clusters: parseInt(numClusters),
      }),
    });

    if (!response.ok) {
      // Try to parse error message from response
      let errorMessage = "Failed to analyze comments";
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch (e) {
        errorMessage = `Server error: ${response.status} ${response.statusText}`;
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();

    // Display results
    displayResults(data);
  } catch (err) {
    console.error("Error:", err);

    // Provide more specific error messages
    let errorMessage = err.message;
    if (err.message === "Failed to fetch" || err.name === "TypeError") {
      errorMessage =
        "Cannot connect to server. Please make sure the backend server is running on http://localhost:5000";
    } else if (
      !errorMessage ||
      errorMessage ===
        "An error occurred while analyzing comments. Please try again."
    ) {
      errorMessage =
        err.message ||
        "An error occurred while analyzing comments. Please try again.";
    }

    showError(errorMessage);
  } finally {
    loading.classList.add("d-none");
  }
}

// Display results
function displayResults(data) {
  // Display video
  displayVideo(data.video);

  // Display statistics
  displayStatistics(data.statistics, data.clusters);

  // Display clusters with comments inside
  displayClusters(data.clusters, data.comments);

  // Show results section
  results.classList.remove("d-none");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Display video
function displayVideo(videoId) {
  const videoContainer = document.getElementById("videoContainer");
  const videoIdDisplay = document.getElementById("videoId");

  videoIdDisplay.textContent = videoId;

  videoContainer.innerHTML = `
    <iframe 
      src="https://www.youtube.com/embed/${videoId}" 
      frameborder="0" 
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
      allowfullscreen>
    </iframe>
  `;
}

// Display statistics
function displayStatistics(statistics, clusters) {
  const statsContainer = document.getElementById("statistics");
  const totalComments = document.getElementById("totalComments");
  const numClustersDisplay = document.getElementById("numClustersDisplay");

  totalComments.textContent = statistics.total_comments;
  numClustersDisplay.textContent = statistics.num_clusters;

  // Calculate sentiment distribution from clusters
  let positiveCount = 0;
  let negativeCount = 0;
  let neutralCount = 0;

  Object.values(clusters).forEach((cluster) => {
    const counts = cluster.sentiment_counts;
    positiveCount += counts.positive || 0;
    negativeCount += counts.negative || 0;
    neutralCount += counts.neutral || 0;
  });

  const total = positiveCount + negativeCount + neutralCount;

  statsContainer.innerHTML = `
    <div class="col-md-4 col-sm-6">
      <div class="stat-card">
        <h3><i class="bi bi-chat-dots"></i> Total Comments</h3>
        <div class="value">${statistics.total_comments}</div>
      </div>
    </div>
    <div class="col-md-4 col-sm-6">
      <div class="stat-card">
        <h3><i class="bi bi-diagram-3"></i> Clusters</h3>
        <div class="value">${statistics.num_clusters}</div>
      </div>
    </div>
    <div class="col-md-4 col-sm-6">
      <div class="stat-card">
        <h3><i class="bi bi-emoji-smile"></i> Positive</h3>
        <div class="value">${positiveCount}</div>
        <div style="font-size: 0.8em; margin-top: 5px;">${
          total > 0 ? ((positiveCount / total) * 100).toFixed(1) : 0
        }%</div>
      </div>
    </div>
    <div class="col-md-4 col-sm-6">
      <div class="stat-card">
        <h3><i class="bi bi-emoji-neutral"></i> Neutral</h3>
        <div class="value">${neutralCount}</div>
        <div style="font-size: 0.8em; margin-top: 5px;">${
          total > 0 ? ((neutralCount / total) * 100).toFixed(1) : 0
        }%</div>
      </div>
    </div>
    <div class="col-md-4 col-sm-6">
      <div class="stat-card">
        <h3><i class="bi bi-emoji-frown"></i> Negative</h3>
        <div class="value">${negativeCount}</div>
        <div style="font-size: 0.8em; margin-top: 5px;">${
          total > 0 ? ((negativeCount / total) * 100).toFixed(1) : 0
        }%</div>
      </div>
    </div>
  `;
}

// Display clusters with comments inside
function displayClusters(clusters, comments) {
  const clustersContainer = document.getElementById("clustersContainer");

  // Group comments by cluster
  const commentsByCluster = {};
  comments.forEach((comment) => {
    const clusterId =
      comment.cluster !== undefined ? comment.cluster : comment.cluster_id;
    if (clusterId !== undefined) {
      // Convert to string to match cluster keys
      const clusterKey = String(clusterId);
      if (!commentsByCluster[clusterKey]) {
        commentsByCluster[clusterKey] = [];
      }
      commentsByCluster[clusterKey].push(comment);
    }
  });

  clustersContainer.innerHTML = Object.entries(clusters)
    .map(([clusterId, cluster]) => {
      const labelClass = cluster.label.replace("_", "-");
      const clusterComments = commentsByCluster[clusterId] || [];

      // Filter comments to only show those matching the cluster's dominant sentiment
      // Use dominant_sentiment which is calculated from the actual sentiment distribution
      const expectedSentiment = cluster.dominant_sentiment;

      const filteredComments = clusterComments.filter((comment) => {
        const commentSentiment = comment.sentiment?.sentiment || "neutral";
        return commentSentiment === expectedSentiment;
      });

      // Render comments for this cluster
      const commentsHTML = filteredComments
        .map((comment) => {
          const sentiment = comment.sentiment?.sentiment || "neutral";
          const sentimentData = comment.sentiment || {};

          const sentimentBadgeClass =
            {
              positive: "bg-success",
              negative: "bg-danger",
              neutral: "bg-secondary",
            }[sentiment] || "bg-secondary";

          return `
            <div class="card comment-card ${sentiment} mb-2" data-sentiment="${sentiment}">
              <div class="card-body p-3">
                <div class="d-flex justify-content-between align-items-start mb-2">
                  <h6 class="card-subtitle mb-0" style="font-size: 0.9rem">
                    <i class="bi bi-person-circle"></i> ${
                      comment.author || comment.authorDisplayName || "Anonymous"
                    }
                  </h6>
                  <span class="badge ${sentimentBadgeClass} comment-sentiment" style="font-size: 0.7rem">${sentiment}</span>
                </div>
                <p class="card-text mb-2" style="font-size: 0.85rem">${
                  comment.text ||
                  comment.cleaned_text ||
                  comment.textDisplay ||
                  "No text"
                }</p>
                <div class="d-flex gap-2 flex-wrap align-items-center">
                  <small class="text-muted" style="font-size: 0.75rem">
                    <i class="bi bi-graph-up"></i> Score: ${
                      sentimentData.compound
                        ? sentimentData.compound.toFixed(3)
                        : "N/A"
                    }
                  </small>
                  ${
                    comment.likes
                      ? `<small class="text-muted" style="font-size: 0.75rem"><i class="bi bi-hand-thumbs-up"></i> ${comment.likes}</small>`
                      : ""
                  }
                </div>
              </div>
            </div>
          `;
        })
        .join("");

      return `
        <div class="col-12 mb-4">
          <div class="card cluster-card ${labelClass} shadow-sm">
            <div class="card-body">
              <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="card-title cluster-title mb-0">${cluster.label.replace(
                  "_",
                  " "
                )}</h5>
                <span class="badge bg-light text-dark">${
                  cluster.count
                } comments</span>
              </div>
              <div class="row g-2 mb-3">
                <div class="col-md-4">
                  <small class="text-muted d-block">Score</small>
                  <strong>${cluster.average_compound}</strong>
                </div>
                <div class="col-md-4">
                  <small class="text-muted d-block">Percentage</small>
                  <strong>${cluster.percentage}%</strong>
                </div>
                <div class="col-md-4">
                  <small class="text-muted d-block">Dominant Sentiment</small>
                  <strong>${cluster.dominant_sentiment}</strong>
                </div>
              </div>
              <div class="mb-3">
                <small class="text-muted d-block mb-2">Sentiment Distribution:</small>
                <div class="d-flex gap-2 flex-wrap">
                  <span class="badge bg-success">Positive: ${
                    cluster.sentiment_counts.positive
                  }</span>
                  <span class="badge bg-secondary">Neutral: ${
                    cluster.sentiment_counts.neutral
                  }</span>
                  <span class="badge bg-danger">Negative: ${
                    cluster.sentiment_counts.negative
                  }</span>
                </div>
              </div>
              <hr class="my-3">
              <h6 class="mb-3">
                <i class="bi bi-chat-left-text"></i> Comments in this cluster 
                <small class="text-muted">(filtered by ${
                  cluster.dominant_sentiment
                } sentiment)</small>:
              </h6>
              <div class="comments-list" style="max-height: 500px; overflow-y: auto;">
                ${
                  commentsHTML ||
                  '<p class="text-muted">No comments matching the cluster sentiment in this cluster.</p>'
                }
              </div>
              ${
                filteredComments.length < clusterComments.length
                  ? `<p class="text-muted mt-2" style="font-size: 0.85rem">
                      <i class="bi bi-info-circle"></i> 
                      Showing ${filteredComments.length} of ${clusterComments.length} comments 
                      (filtered to show only ${cluster.dominant_sentiment} sentiment)
                    </p>`
                  : ""
              }
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}

// Display comments
function displayComments(comments) {
  const commentsContainer = document.getElementById("commentsContainer");

  commentsContainer.innerHTML = comments
    .map((comment) => {
      const sentiment = comment.sentiment?.sentiment || "neutral";
      const sentimentData = comment.sentiment || {};
      const clusterLabel =
        comment.cluster_label || `Cluster ${comment.cluster}`;

      const sentimentBadgeClass =
        {
          positive: "bg-success",
          negative: "bg-danger",
          neutral: "bg-secondary",
        }[sentiment] || "bg-secondary";

      return `
        <div class="card comment-card ${sentiment} mb-3" data-sentiment="${sentiment}">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-start mb-2">
              <h6 class="card-subtitle mb-0">
                <i class="bi bi-person-circle"></i> ${
                  comment.author || "Anonymous"
                }
              </h6>
              <span class="badge ${sentimentBadgeClass} comment-sentiment">${sentiment}</span>
            </div>
            <p class="card-text">${
              comment.text || comment.cleaned_text || "No text"
            }</p>
            <div class="d-flex gap-3 flex-wrap align-items-center">
              <small class="text-muted">
                <i class="bi bi-diagram-3"></i> Cluster: 
                <span class="badge bg-primary">${clusterLabel}</span>
              </small>
              <small class="text-muted">
                <i class="bi bi-graph-up"></i> Score: ${
                  sentimentData.compound || "N/A"
                }
              </small>
              ${
                comment.likes
                  ? `<small class="text-muted"><i class="bi bi-hand-thumbs-up"></i> ${comment.likes}</small>`
                  : ""
              }
            </div>
          </div>
        </div>
      `;
    })
    .join("");

  // Setup filter functionality
  setupFilters();
}

// Setup filter buttons
function setupFilters() {
  const filterButtons = document.querySelectorAll(".filter-btn");
  const commentCards = document.querySelectorAll(".comment-card");

  filterButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      // Update active button
      filterButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      // Filter comments
      const filter = btn.dataset.filter;
      commentCards.forEach((card) => {
        if (filter === "all") {
          card.classList.remove("d-none");
        } else {
          const sentiment = card.dataset.sentiment;
          if (sentiment === filter) {
            card.classList.remove("d-none");
          } else {
            card.classList.add("d-none");
          }
        }
      });
    });
  });
}

// Event listeners
analyzeBtn.addEventListener("click", analyzeComments);

videoUrlInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    analyzeComments();
  }
});

numClustersInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    analyzeComments();
  }
});

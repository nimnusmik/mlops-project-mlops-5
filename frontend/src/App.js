import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [latestRecommendations, setLatestRecommendations] = useState([]);
  const [expanded, setExpanded] = useState({}); // 줄거리 토글 상태

  // 줄거리 토글
  const toggleOverview = (contentId) => {
    setExpanded((prev) => ({
      ...prev,
      [contentId]: !prev[contentId],
    }));
  };

  // 최신 추천 영화 불러오기
  useEffect(() => {
    console.log("API ENDPOINT:", process.env.REACT_APP_API_ENDPOINT);

    fetch(`${process.env.REACT_APP_API_ENDPOINT}/latest-recommendations`)
      .then(res => res.json())
      .then(data => {
        setLatestRecommendations(data.recent_recommendations || []);
      });
  }, []);

  // 추천 카드 렌더링
  const renderCards = (recommendations) => (
    <div className="recommendation-grid">
      {recommendations.map((item) => (
        <div
          className="card"
          key={item.content_id}
          onClick={() => toggleOverview(item.content_id)}
        >
          <img className="poster" src={item.poster_url} alt={item.title} />
          <div className="title">{item.title}</div>
          {expanded[item.content_id] && (
            <div className="overview">{item.overview || "줄거리 없음"}</div>
          )}
        </div>
      ))}
    </div>
  );

  return (
    <div className="App">
      <h1>🎬 영화 추천</h1>
      {renderCards(latestRecommendations)}
    </div>
  );
}

export default App;

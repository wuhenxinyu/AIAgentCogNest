import { useState, useEffect } from 'react';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 从本地存储加载历史记录
    const loadHistory = () => {
      try {
        const savedHistory = localStorage.getItem('qaHistory');
        if (savedHistory) {
          setHistory(JSON.parse(savedHistory));
        }
      } catch (error) {
        console.error('加载历史记录失败:', error);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  const handleClearHistory = () => {
    if (window.confirm('确定要清空所有历史记录吗？')) {
      localStorage.removeItem('qaHistory');
      setHistory([]);
    }
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  if (loading) {
    return <div className="history">加载中...</div>;
  }

  return (
    <div className="history">
      <div className="history-header">
        <h2>查询历史</h2>
        {history.length > 0 && (
          <button className="btn btn-danger" onClick={handleClearHistory}>
            清空历史记录
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="no-history">
          <p>暂无查询历史</p>
        </div>
      ) : (
        <div className="history-list">
          {history.map((item, index) => (
            <div key={index} className="history-item">
              <div className="history-meta">
                <span className="history-type">
                  {item.batch ? '批量查询' : '单个查询'}
                </span>
                <span className="history-time">
                  {formatDate(item.timestamp)}
                </span>
              </div>

              {item.batch ? (
                // 批量查询记录
                <div className="batch-history">
                  <div className="batch-questions">
                    <h4>问题列表:</h4>
                    <ul>
                      {item.questions.map((q, qIndex) => (
                        <li key={qIndex}>{q}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="batch-answers">
                    <h4>答案列表:</h4>
                    {item.answers.map((answer, aIndex) => (
                      <div key={aIndex} className="batch-answer">
                        <p><strong>答案 {aIndex + 1}:</strong> {answer.answer || '未找到答案'}</p>
                        {answer.error && (
                          <p className="batch-answer-error">错误: {answer.error}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                // 单个查询记录
                <div className="single-history">
                  <div className="history-question">
                    <h4>问题:</h4>
                    <p>{item.question}</p>
                  </div>
                  <div className="history-answer">
                    <h4>答案:</h4>
                    <p>{item.answer.answer || '未找到答案'}</p>
                  </div>
                  {item.answer.references && item.answer.references.length > 0 && (
                    <div className="history-references">
                      <h4>参考来源:</h4>
                      <ul>
                        {item.answer.references.map((ref, refIndex) => (
                          <li key={refIndex}>
                            PDF SHA1: {ref.pdf_sha1}, 页码: {ref.page_index}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default History;
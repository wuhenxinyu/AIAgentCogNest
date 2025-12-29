import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="home">
      <div className="hero-section">
        <div className="hero-content">
          {/* 左侧文本内容区域 */}
          <div className="hero-text">
            <h1 className="hero-title">
              RAG Challenge <span className="hero-highlight">智能</span>
            </h1>
            
            <h2 className="hero-title"><span className="hero-highlight">问答系统</span></h2>
            
            <p className="hero-description">
              基于检索增强生成(RAG)技术的企业年度报告智能问答系统，能
              够快速准确地回答关于公司财务、运营和战略的问题。
            </p>
            <div className="hero-buttons">
              <Link to="/demo" className="hero-button primary">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M8 5V19L19 12L8 5Z" fill="#6366F1"/>
                </svg>
                开始体验
              </Link>
              <Link to="/analytics" className="hero-button secondary">
                查看分析
              </Link>
            </div>
          </div>
          
          {/* 右侧功能图标区域 */}
          <div className="hero-features-right">
            <div className="hero-feature ai-answer-feature">
              <div className="feature-icon ai-answer">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 15C10.34 15 9 13.66 9 12C9 10.34 10.34 9 12 9C13.66 9 15 10.34 15 12C15 13.66 13.66 15 12 15ZM16.94 17.85C15.73 17.31 14.08 17.5 13.03 18.43L10.97 16.37C11.8 15.32 12.03 13.68 11.5 12.47C10.97 11.27 9.33 10.96 8.13 11.5L6.07 9.44C7.27 8.9 8.91 9.21 9.96 10.26C11 11.3 10.69 12.94 10.14 14.14C9.59 15.34 8.25 15.65 7.15 15.1L5.09 13.04C6.67 11.79 7.32 9.64 6.82 7.74C6.31 5.84 4.68 4.31 2.78 3.8C0.88 3.29 -0.65 1.66 -1.16 0C0.33 -1.17 2.48 -1.82 4.38 -1.32C6.28 -0.82 7.81 0.81 8.32 2.71C8.83 4.61 8.18 6.76 6.5 8.01L8.56 10.07C10.14 8.82 10.79 6.67 10.29 4.77C9.78 2.87 8.15 1.34 6.25 0.83C4.35 0.32 2.2 0.97 0.95 2.55L0 3.55C1.49 5.55 2.14 7.9 1.63 10C1.12 12.1 0.47 14.45 1.96 16.45C3.45 18.45 5.8 19.1 7.9 18.59C10 18.08 12.35 17.43 14.35 18.92L15.35 18.03C13.61 16.86 12.96 14.91 13.47 13.01C13.98 11.11 15.61 9.58 17.51 9.07C19.41 8.56 21.36 9.21 22.53 10.95L23.53 10C22.04 8 21.39 5.65 21.9 3.55C22.41 1.45 24.04 -0.12 25.94 0.39C27.84 0.9 29.37 2.53 29.88 4.43C30.39 6.33 29.74 8.48 28.06 9.73L26.01 11.78C27.59 13.03 28.24 15.18 27.73 17.08C27.22 18.98 25.59 20.51 23.69 21.02C21.79 21.53 19.64 20.88 18.39 19.3L16.94 17.85Z" fill="white"/>
                </svg>
              </div>
              <div className="feature-label">AI问答</div>
            </div>
            <div className="hero-feature pdf-parse-feature">
              <div className="feature-icon pdf-parse">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V8L14 2ZM18 20H6V4H13V9H18V20ZM12 13H8V15H12V13ZM16 13H14V15H16V13ZM12 17H8V19H12V17ZM16 17H14V19H16V17Z" fill="white"/>
                </svg>
              </div>
              <div className="feature-label">PDF解析</div>
            </div>
            <div className="hero-feature smart-search-feature">
              <div className="feature-icon smart-search">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3C5.91 3 3 5.91 3 9.5C3 13.09 5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5C5 7.01 7.01 5 9.5 5C11.99 5 14 7.01 14 9.5C14 11.99 11.99 14 9.5 14Z" fill="white"/>
                </svg>
              </div>
              <div className="feature-label">智能检索</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
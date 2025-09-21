-- 股票历史数据表
-- 字段说明：
-- stock_name：股票名称，字符串
-- ts_code：股票代码，字符串
-- trade_date：交易日期，日期
-- open：开盘价，浮点数
-- high：最高价，浮点数
-- low：最低价，浮点数
-- close：收盘价，浮点数
-- vol：成交量，浮点数
-- amount：成交额，浮点数

CREATE TABLE stock_price (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    stock_name VARCHAR(20) NOT NULL COMMENT '股票名称',
    ts_code VARCHAR(20) NOT NULL COMMENT '股票代码',
    trade_date VARCHAR(10) NOT NULL COMMENT '交易日期',
    open DECIMAL(15,2) COMMENT '开盘价',
    high DECIMAL(15,2) COMMENT '最高价',
    low DECIMAL(15,2) COMMENT '最低价',
    close DECIMAL(15,2) COMMENT '收盘价',
    vol DECIMAL(20,2) COMMENT '成交量',
    amount DECIMAL(20,2) COMMENT '成交额',
    UNIQUE KEY uniq_stock_date (ts_code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票历史价格数据表'; 
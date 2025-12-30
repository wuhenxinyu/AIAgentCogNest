import React, { useState } from 'react';
import { Send, Play, MessageCircle, Database, BarChart3 } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { api, chatHelper } from '../services/api';

interface DataSource {
  name: string;
  table: string;
  rows: number;
  columns: string[];
  description: string;
  source: string;
}

interface QueryMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  data?: any[];
  timestamp: Date;
}

interface QueryPanelProps {
  selectedSource: DataSource | null;
  onQueryResult: (result: any) => void;
}

export function QueryPanelWithDataSources({
  selectedSource,
  onQueryResult
}: QueryPanelProps) {
  const [query, setQuery] = useState('');
  const [chatMessages, setChatMessages] = useState<QueryMessage[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('query');

  // 根据数据源生成快速问题
  const getQuickQuestions = () => {
    if (!selectedSource) return [];

    const questions = [
      '显示前10条数据',
      '总记录数是多少？',
      '数据概览'
    ];

    // 根据数据源类型添加特定问题
    if (selectedSource.table === 'sales_data') {
      questions.push(
        '各类别产品销量排名',
        '销售额最高的品牌',
        '平均折扣率分析'
      );
    } else if (selectedSource.table === 'erp_products') {
      questions.push(
        '价格最高的产品',
        '库存不足的产品',
        '各品类产品分布'
      );
    } else if (selectedSource.table === 'erp_orders') {
      questions.push(
        '最近7天的订单',
        '订单状态统计',
        '客单价分析'
      );
    } else if (selectedSource.name.includes('销售')) {
      questions.push(
        '月度销售趋势',
        '销售业绩排名',
        '同比/环比增长'
      );
    }

    return questions;
  };

  const handleQuery = async () => {
    if (!query.trim() || !selectedSource) return;

    setIsLoading(true);
    const userMessage: QueryMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
      timestamp: new Date()
    };
    setChatMessages(prev => [...prev, userMessage]);

    try {
      let result;

      if (activeTab === 'query') {
        // 直接查询
        const queryData = {
          query: query,
          table_name: selectedSource.table,
          limit: 100
        };

        const response = await api.queryData(queryData);
        result = {
          success: response.success,
          data: response.data || [],
          answer: response.answer || '查询成功',
          columns: response.columns || [],
          totalRows: response.total_rows || 0,
          returnedRows: response.returned_rows || 0,
          source: response.source
        };
      } else {
        // 聊天模式
        result = await chatHelper(query, undefined, sessionId, selectedSource.table);
        if (!sessionId) {
          setSessionId(result.sessionId);
        }
      }

      if (result.success) {
        const assistantMessage: QueryMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: result.answer || '查询成功',
          data: result.data,
          timestamp: new Date()
        };
        setChatMessages(prev => [...prev, assistantMessage]);
        onQueryResult(result);
      } else {
        const errorMessage: QueryMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `查询失败：${result.error || '未知错误'}`,
          timestamp: new Date()
        };
        setChatMessages(prev => [...prev, errorMessage]);
      }
    } catch (error: any) {
      const errorMessage: QueryMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `查询失败：${error.message || '网络错误'}`,
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setQuery('');
    }
  };

  const handleQuickQuestion = (question: string) => {
    setQuery(question);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">数据查询</CardTitle>
          {selectedSource && (
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                {selectedSource.source === 'csv' ? '真实数据' :
                 selectedSource.source === 'upload' ? '上传文件' :
                 selectedSource.source === 'mock' ? '模拟数据' : '汇总数据'}
              </Badge>
              <span className="text-sm text-gray-500">
                {selectedSource.name}
              </span>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="h-full pt-0">
        {selectedSource ? (
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="query" className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                直接查询
              </TabsTrigger>
              <TabsTrigger value="chat" className="flex items-center gap-2">
                <MessageCircle className="w-4 h-4" />
                智能对话
              </TabsTrigger>
            </TabsList>

            {/* 数据源信息 */}
            <div className="mt-3 p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2 text-sm">
                <Database className="w-4 h-4 text-gray-500" />
                <span className="font-medium">{selectedSource.name}</span>
                <Badge variant="secondary" className="text-xs">
                  {selectedSource.rows.toLocaleString()} 行
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {selectedSource.columns.length} 列
                </Badge>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {selectedSource.description}
              </p>
            </div>

            {/* 聊天历史区域 */}
            {chatMessages.length > 0 && (
              <ScrollArea className="flex-1 mt-4 border rounded-lg p-4 bg-gray-50">
                <div className="space-y-4">
                  {chatMessages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg px-3 py-2 ${
                          msg.role === 'user'
                            ? 'bg-blue-500 text-white'
                            : 'bg-gray-200 text-gray-800'
                        }`}
                      >
                        <p className="text-sm">{msg.content}</p>
                        <p className={`text-xs mt-1 ${
                          msg.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                        }`}>
                          {formatTime(msg.timestamp)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}

            {/* 查询区域 */}
            <div className="mt-4 space-y-3">
              <Textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  activeTab === 'query'
                    ? '输入查询条件，例如：显示销售额最高的10条记录'
                    : '用自然语言描述您的需求，例如：帮我分析一下销售趋势'
                }
                className="min-h-[80px] resize-none"
                disabled={isLoading}
              />

              {/* 快速问题 */}
              <div className="space-y-2">
                <p className="text-xs text-gray-500">快速选择：</p>
                <div className="flex flex-wrap gap-2">
                  {getQuickQuestions().map((q, idx) => (
                    <Button
                      key={idx}
                      variant="outline"
                      size="sm"
                      onClick={() => handleQuickQuestion(q)}
                      disabled={isLoading}
                      className="text-xs"
                    >
                      {q}
                    </Button>
                  ))}
                </div>
              </div>

              {/* 提交按钮 */}
              <Button
                onClick={handleQuery}
                disabled={!query.trim() || isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    处理中...
                  </>
                ) : (
                  <>
                    {activeTab === 'query' ? (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        执行查询
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4 mr-2" />
                        发送消息
                      </>
                    )}
                  </>
                )}
              </Button>
            </div>
          </Tabs>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Database className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">请先选择一个数据源</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
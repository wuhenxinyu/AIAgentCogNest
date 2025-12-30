import React, { useState } from 'react';
import { Send, Play, MessageCircle, BarChart3, History } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { queryDataHelper, chatHelper } from '../services/api';

interface QueryMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  data?: any[];
  timestamp: Date;
}

interface NewQueryPanelProps {
  selectedFileId?: string;
  onQueryResult: (result: any) => void;
  onVisualizationRequest: (data: any) => void;
}

export function NewQueryPanel({
  selectedFileId,
  onQueryResult,
  onVisualizationRequest
}: NewQueryPanelProps) {
  const [query, setQuery] = useState('');
  const [chatMessages, setChatMessages] = useState<QueryMessage[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('query');

  // 常见问题模板
  const quickQuestions = [
    '显示前10条数据',
    '统计各列的数量',
    '计算平均值',
    '查找最大值和最小值',
    '分组统计',
    '排序数据',
    '筛选特定条件的数据'
  ];

  const handleQuery = async () => {
    if (!query.trim() || !selectedFileId) return;

    setIsLoading(true);
    const userMessage: QueryMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
      timestamp: new Date()
    };
    setChatMessages(prev => [...prev, userMessage]);

    try {
      const result = await queryDataHelper(query, selectedFileId);

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
          content: `查询失败：${result.error}`,
          timestamp: new Date()
        };
        setChatMessages(prev => [...prev, errorMessage]);
      }
    } catch (error: any) {
      const errorMessage: QueryMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `查询失败：${error.message}`,
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setQuery('');
    }
  };

  const handleChat = async () => {
    if (!query.trim() || !selectedFileId) return;

    setIsLoading(true);
    const userMessage: QueryMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
      timestamp: new Date()
    };
    setChatMessages(prev => [...prev, userMessage]);

    try {
      const result = await chatHelper(query, selectedFileId, sessionId);

      if (result.success) {
        if (!sessionId) {
          setSessionId(result.sessionId);
        }

        const assistantMessage: QueryMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: result.message,
          data: result.data,
          timestamp: new Date()
        };
        setChatMessages(prev => [...prev, assistantMessage]);

        if (result.data && result.data.length > 0) {
          onQueryResult({
            success: true,
            data: result.data,
            columns: Object.keys(result.data[0]),
            totalRows: result.data.length,
            answer: result.message
          });
        }
      } else {
        const errorMessage: QueryMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `回复失败：${result.error}`,
          timestamp: new Date()
        };
        setChatMessages(prev => [...prev, errorMessage]);
      }
    } catch (error: any) {
      const errorMessage: QueryMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `回复失败：${error.message}`,
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setQuery('');
    }
  };

  const handleSubmit = () => {
    if (activeTab === 'query') {
      handleQuery();
    } else {
      handleChat();
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

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">数据查询</CardTitle>
          {!selectedFileId && (
            <Badge variant="destructive" className="text-xs">
              请先上传文件
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="h-full pt-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="query" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              直接查询
            </TabsTrigger>
            <TabsTrigger value="chat" className="flex items-center gap-2">
              <MessageCircle className="w-4 h-4" />
              对话分析
            </TabsTrigger>
          </TabsList>

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
              placeholder={
                activeTab === 'query'
                  ? '输入查询条件，例如：显示销售额最高的10条记录'
                  : '用自然语言描述您的需求，例如：帮我分析一下销售趋势'
              }
              className="min-h-[80px] resize-none"
              disabled={!selectedFileId || isLoading}
            />

            {/* 快速问题 */}
            <div className="space-y-2">
              <p className="text-xs text-gray-500">快速选择：</p>
              <div className="flex flex-wrap gap-2">
                {quickQuestions.map((q, idx) => (
                  <Button
                    key={idx}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuickQuestion(q)}
                    disabled={!selectedFileId}
                    className="text-xs"
                  >
                    {q}
                  </Button>
                ))}
              </div>
            </div>

            {/* 提交按钮 */}
            <Button
              onClick={handleSubmit}
              disabled={!query.trim() || !selectedFileId || isLoading}
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
      </CardContent>
    </Card>
  );
}
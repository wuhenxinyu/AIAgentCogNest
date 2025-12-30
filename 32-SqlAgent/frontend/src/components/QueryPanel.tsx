import React, { useState, useEffect } from 'react';
import { Send, Play, FileText, Lightbulb, ChevronDown, ChevronRight } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { api } from '../services/api';

interface QueryPanelProps {
  selectedTable?: string | null;
  onQueryResult?: (result: any) => void;
}

export function QueryPanel({ selectedTable, onQueryResult }: QueryPanelProps) {
  const [query, setQuery] = useState('显示前10条数据');
  const [showSQL, setShowSQL] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [generatedSQL, setGeneratedSQL] = useState('');
  const [reasoning, setReasoning] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);

  // 示例问题模板（从简单到复杂）
  const exampleQuestions = [
    '销售额最高的前10个产品',
    '好评率超过95%且销量过万的产品',
    '各品牌在智能手机分类中的销量对比',
    '折扣率>30%且价格<5000的性价比产品'
  ];

  // 当选中的表变化时，更新提示
  useEffect(() => {
    if (selectedTable) {
      // 如果是上传的文件（file_开头），提取 file_id
      if (selectedTable.startsWith('file_')) {
        setSelectedFileId(selectedTable.replace('file_', ''));
        setQuery(`显示前10条数据`);
      } else {
        setSelectedFileId(null);
        setQuery(`显示${selectedTable}表的前10条数据`);
      }
    }
  }, [selectedTable]);

  const handleRun = async () => {
    if (!selectedTable) {
      alert('请先在左侧选择一个数据表或上传文件');
      return;
    }

    if (!query.trim()) {
      alert('请输入查询问题');
      return;
    }

    console.log('='.repeat(80));
    console.log('[CSV查询] 开始查询:', { query, selectedTable, selectedFileId });
    console.log('[CSV查询] 是否为CSV文件:', !!selectedFileId);
    setIsQuerying(true);
    try {
      // 调用后端查询API
      const request: any = {
        query: query
        // 不传递固定的 limit，让 LLM 根据用户问题决定
      };
      
      // 如果是上传的文件，传递 file_id；否则传递 table_name
      if (selectedFileId) {
        request.file_id = selectedFileId;
        console.log('[CSV查询] 发送请求 - file_id:', selectedFileId, 'query:', query);
      } else {
        request.table_name = selectedTable;
        console.log('[数据库查询] 发送请求 - table_name:', selectedTable, 'query:', query);
      }
      
      const result = await api.queryData(request);

      console.log('[查询结果] SQL:', result.sql);
      console.log('[查询结果] 数据行数:', result.data?.length);
      console.log('[查询结果] 答案长度:', result.answer?.length);
      console.log('='.repeat(80));

      // 如果返回了SQL，显示它
      if (result.sql) {
        setGeneratedSQL(result.sql);
        setShowSQL(true);
      }

      // 如果返回了推理步骤，显示它
      if (result.reasoning && Array.isArray(result.reasoning)) {
        setReasoning(result.reasoning);
        setShowReasoning(true);
      }

      // 传递结果给父组件
      if (onQueryResult) {
        onQueryResult(result);
      }
    } catch (error) {
      console.error('查询失败:', error);
      alert('查询失败: ' + error);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#0F1123] overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-white/5 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-cyan-400 font-medium">智能问答区</h2>
          {selectedTable && (
            <span className="text-xs text-gray-500">
              当前表: <span className="text-cyan-400">{selectedTable}</span>
            </span>
          )}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="px-6 py-4 space-y-4">
          {/* SQL Section */}
          {generatedSQL && (
            <Collapsible open={showSQL} onOpenChange={setShowSQL}>
              <CollapsibleTrigger className="flex items-center gap-2 w-full px-4 py-3 bg-[#13152E] rounded-lg border border-purple-500/20 hover:border-purple-500/40 transition-colors">
                {showSQL ? (
                  <ChevronDown className="w-4 h-4 text-purple-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-purple-400" />
                )}
                <FileText className="w-4 h-4 text-purple-400" />
                <span className="text-purple-300 text-sm font-medium">查看 SQL</span>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="bg-[#0B0D1E] rounded-lg border border-white/5 p-4 font-mono text-xs overflow-x-auto max-h-80 overflow-y-auto">
                  <pre className="text-gray-300 leading-relaxed whitespace-pre-wrap">
                    {generatedSQL}
                  </pre>
                </div>
              </CollapsibleContent>
            </Collapsible>
          )}

          {/* Reasoning Section */}
          {reasoning.length > 0 && (
            <Collapsible open={showReasoning} onOpenChange={setShowReasoning}>
              <CollapsibleTrigger className="flex items-center gap-2 w-full px-4 py-3 bg-[#13152E] rounded-lg border border-cyan-500/20 hover:border-cyan-500/40 transition-colors">
                {showReasoning ? (
                  <ChevronDown className="w-4 h-4 text-cyan-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-cyan-400" />
                )}
                <Lightbulb className="w-4 h-4 text-cyan-400" />
                <span className="text-cyan-300 text-sm font-medium">生成思路</span>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="bg-[#0B0D1E] rounded-lg border border-white/5 p-4">
                  <div className="space-y-2">
                    {reasoning.map((step, idx) => (
                      <div key={idx} className="flex gap-3 text-xs">
                        <span className="text-cyan-400 font-mono flex-shrink-0">{idx + 1}.</span>
                        <span className="text-gray-300">{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-white/5 p-6 flex-shrink-0">
        <div className="space-y-3">
          {/* 示例问答 - 移到输入框上方 */}
          <div>
            <p className="text-xs text-gray-500 mb-3">💡 示例问答</p>
            <div className="grid grid-cols-2 gap-2">
              {exampleQuestions.map((question, idx) => (
                <Button
                  key={idx}
                  size="sm"
                  onClick={() => setQuery(question)}
                  disabled={!selectedTable}
                  className="text-xs bg-[#1a1d3e] hover:bg-[#252850] text-gray-300 border border-white/10 justify-start transition-colors"
                >
                  {question}
                </Button>
              ))}
            </div>
          </div>

          <Textarea
            placeholder={selectedTable ? `输入您的问题，例如：${selectedTable}表中销量最高的产品是什么？` : "请先在左侧选择一个数据表..."}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={!selectedTable}
            className="min-h-[100px] bg-[#13152E] border-white/10 text-gray-300 placeholder-gray-600 resize-none focus:border-cyan-500/30"
          />

          <div className="flex justify-end items-center">
            <Button
              onClick={handleRun}
              disabled={!selectedTable || !query.trim() || isQuerying}
              className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/20"
            >
              {isQuerying ? (
                <>正在查询...</>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  运行查询
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect } from 'react';
import { Download, RefreshCw, BarChart3, Table as TableIcon, FileText } from 'lucide-react';
import { Button } from './ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { api } from '../services/api';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Hash } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ResultsPanelProps {
  queryResult?: any;
}

export function ResultsPanel({ queryResult: externalResult }: ResultsPanelProps) {
  const [queryResult, setQueryResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [chartType, setChartType] = useState<'bar' | 'line'>('bar');

  useEffect(() => {
    if (externalResult) {
      setQueryResult(externalResult);
    }
    // 不再自动加载默认数据，等待用户选择表格并查询
  }, [externalResult]);

  const loadDefaultData = async () => {
    setLoading(true);
    try {
      const result = await api.queryData({
        query: '显示前10条数据',
        table_name: 'sales_data',
        limit: 10
      });
      setQueryResult(result);
    } catch (error) {
      console.error('加载默认数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    if (externalResult) {
      setQueryResult(externalResult);
    } else {
      loadDefaultData();
    }
  };

  const handleExport = () => {
    if (!queryResult || !queryResult.data) return;

    const headers = queryResult.columns?.join(',') || '';
    const rows = queryResult.data.map((row: any) =>
      Object.values(row).join(',')
    ).join('\n');

    const csv = `${headers}\n${rows}`;
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `query_result_${Date.now()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (loading) {
    return (
      <div className="h-full flex flex-col bg-[#0B0D1E] overflow-hidden">
        <div className="px-4 py-4 border-b border-white/5 flex-shrink-0">
          <h2 className="text-cyan-400 font-medium text-sm">查询结果</h2>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-gray-400">加载中...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-[#0B0D1E] overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-white/5 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <TableIcon className="w-5 h-5 text-cyan-400" />
            <h2 className="text-cyan-400 text-base font-medium">查询结果与图表</h2>
            {queryResult && (
              <div className="flex items-center gap-3 text-xs text-gray-400 ml-2">
                <span>共 {queryResult.returned_rows || queryResult.data?.length || 0} 行</span>
                <span>执行时间: {queryResult.executionTime || '125'}ms</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              className="h-8 px-3 text-gray-400 hover:text-cyan-400"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleExport}
              disabled={!queryResult || !queryResult.data}
              className="h-8 px-3 text-gray-400 hover:text-cyan-400"
            >
              <Download className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto min-h-0 px-4 py-4">
        {!queryResult || !queryResult.data || queryResult.data.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-gray-600 mx-auto mb-3" />
              <p className="text-gray-500 text-sm">暂无数据</p>
              <p className="text-gray-600 text-xs mt-1">请在左侧选择数据源并执行查询</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Data Table */}
            <div className="bg-[#13152E] rounded-lg border border-white/5 overflow-hidden">
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/5 hover:bg-transparent">
                      {(queryResult.columns || Object.keys(queryResult.data[0])).map((col: string, idx: number) => (
                        <TableHead key={idx} className="text-cyan-400 font-medium text-xs">
                          {col}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {queryResult.data.map((row: any, rowIdx: number) => (
                      <TableRow key={rowIdx} className="border-white/5 hover:bg-white/5">
                        {(queryResult.columns || Object.keys(row)).map((col: string, colIdx: number) => (
                          <TableCell key={colIdx} className="text-gray-300 text-xs">
                            {row[col] !== null && row[col] !== undefined ? String(row[col]) : '-'}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>

            {/* Auto-generated Chart */}
            {queryResult.data && queryResult.data.length > 0 && (() => {
              const columns = queryResult.columns || Object.keys(queryResult.data[0]);
              // 找到第一个文本列作为 X 轴
              const xKey = columns.find((col: string) => 
                typeof queryResult.data[0][col] === 'string'
              ) || columns[0];
              
              // 找到第一个数字列作为 Y 轴
              const yKey = columns.find((col: string) => 
                typeof queryResult.data[0][col] === 'number' && col !== xKey
              ) || columns.find((col: string) => col !== xKey) || columns[1];

              // 字段名翻译映射
              const fieldNameMap: Record<string, string> = {
                'product_name': '产品名称',
                'price': '价格',
                'sales_volume': '销售数量',
                'sale_date': '销售日期',
                'category': '类别',
                'brand': '品牌',
                'total_amount': '销售总额',
                'quantity': '数量',
                'name': '名称',
              };

              const getFieldLabel = (field: string) => fieldNameMap[field] || field;

              // 图表数据：如果数据量较少（<=20条），显示全部；否则只显示前20条避免拥挤
              const chartData = queryResult.data.length <= 20 
                ? queryResult.data 
                : queryResult.data.slice(0, 20);

              return (
                <div className="bg-[#13152E] rounded-lg border border-white/5 p-4">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-sm font-medium text-cyan-400">自动生成的图表</h3>
                      {queryResult.data.length > 20 && (
                        <span className="text-xs text-gray-500">（显示前20条）</span>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        onClick={() => setChartType('bar')}
                        className={`text-xs h-7 ${
                          chartType === 'bar'
                            ? 'bg-purple-600 hover:bg-purple-700 text-white'
                            : 'bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 border border-purple-500/40'
                        }`}
                      >
                        柱状图
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => setChartType('line')}
                        className={`text-xs h-7 ${
                          chartType === 'line'
                            ? 'bg-pink-600 hover:bg-pink-700 text-white'
                            : 'bg-pink-600/20 hover:bg-pink-600/30 text-pink-300 border border-pink-500/40'
                        }`}
                      >
                        折线图
                      </Button>
                    </div>
                  </div>
                  <div className="flex gap-2 mb-3">
                    <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#1e2139] border border-[#2a2d4a]">
                      <Hash className="w-3 h-3 text-cyan-400" />
                      <span className="text-xs font-medium text-gray-300">维度: {getFieldLabel(xKey)}</span>
                    </div>
                    <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#1e2139] border border-[#2a2d4a]">
                      <Hash className="w-3 h-3 text-cyan-400" />
                      <span className="text-xs font-medium text-gray-300">指标: {getFieldLabel(yKey)}</span>
                    </div>
                  </div>
                  <ResponsiveContainer width="100%" height={250}>
                    {chartType === 'bar' ? (
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis 
                          dataKey={xKey} 
                          stroke="#6ee7b7" 
                          tick={{ fill: '#9ca3af', fontSize: 11 }}
                          angle={-15}
                          textAnchor="end"
                          height={60}
                        />
                        <YAxis 
                          stroke="#6ee7b7" 
                          tick={{ fill: '#9ca3af', fontSize: 11 }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1a1b3e', 
                            border: '1px solid #5ce1e6',
                            borderRadius: '8px',
                            fontSize: '12px'
                          }}
                        />
                        <Legend 
                          wrapperStyle={{ fontSize: '12px', color: '#9ca3af' }}
                        />
                        <Bar dataKey={yKey} fill="#22d3ee" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    ) : (
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis 
                          dataKey={xKey} 
                          stroke="#6ee7b7" 
                          tick={{ fill: '#9ca3af', fontSize: 11 }}
                          angle={-15}
                          textAnchor="end"
                          height={60}
                        />
                        <YAxis 
                          stroke="#6ee7b7" 
                          tick={{ fill: '#9ca3af', fontSize: 11 }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1a1b3e', 
                            border: '1px solid #5ce1e6',
                            borderRadius: '8px',
                            fontSize: '12px'
                          }}
                        />
                        <Legend 
                          wrapperStyle={{ fontSize: '12px', color: '#9ca3af' }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey={yKey} 
                          stroke="#ec4899" 
                          strokeWidth={2}
                          dot={{ fill: '#ec4899', r: 4 }}
                        />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </div>
              );
            })()}

            {/* Analysis Report */}
            {queryResult.answer && queryResult.answer.trim() && (
              <div className="bg-[#13152E] rounded-lg border border-white/5 p-5">
                <div className="flex items-center gap-2 mb-4 pb-3 border-b border-white/5">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-medium text-cyan-400">数据分析报告</h3>
                </div>
                <div className="prose prose-sm prose-invert max-w-none">
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h2: ({node, ...props}) => <h2 className="text-base font-semibold text-cyan-300 mt-4 mb-2" {...props} />,
                      h3: ({node, ...props}) => <h3 className="text-sm font-medium text-purple-300 mt-3 mb-2" {...props} />,
                      p: ({node, ...props}) => <p className="text-gray-300 text-xs leading-relaxed mb-2" {...props} />,
                      ul: ({node, ...props}) => <ul className="text-gray-300 text-xs space-y-1 ml-4 mb-2" {...props} />,
                      li: ({node, ...props}) => <li className="text-gray-300" {...props} />,
                      strong: ({node, ...props}) => <strong className="text-cyan-400 font-medium" {...props} />,
                      code: ({node, ...props}) => <code className="text-purple-300 bg-purple-500/10 px-1 py-0.5 rounded text-xs" {...props} />,
                    }}
                  >
                    {queryResult.answer}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            {/* Stats */}
            <div className="flex items-center gap-4 text-xs text-gray-500">
              <span>总行数: {queryResult.total_rows || queryResult.data.length}</span>
              <span>返回行数: {queryResult.returned_rows || queryResult.data.length}</span>
              {queryResult.source && <span>数据源: {queryResult.source}</span>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

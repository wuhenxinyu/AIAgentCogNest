import React, { useState } from 'react';
import { DataSourcePanel } from './components/DataSourcePanel';
import { QueryPanel } from './components/QueryPanel';
import { ResultsPanel } from './components/ResultsPanel';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';

export default function App() {
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [queryResult, setQueryResult] = useState<any>(null);

  const handleTableSelect = (tableName: string) => {
    setSelectedTable(tableName);
    console.log('选中表:', tableName);
  };

  const handleQueryResult = (result: any) => {
    setQueryResult(result);
    if (result.success) {
      toast.success('查询执行成功', {
        description: `返回 ${result.returned_rows || result.data?.length || 0} 条结果`,
      });
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-[#0B0D1E] text-gray-100">
      {/* Header */}
      <div className="h-16 border-b border-white/5 bg-[#0B0D1E] flex items-center px-6 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h1 className="text-lg font-semibold">
            <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
              智能问数
            </span>
          </h1>
        </div>
        <div className="ml-auto flex items-center gap-4">
          {selectedTable && (
            <div className="text-sm text-gray-400">
              当前表: <span className="text-cyan-400">{selectedTable}</span>
            </div>
          )}
        </div>
      </div>

      {/* Main Content - Three Column Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Data Sources */}
        <div className="w-[230px] flex-shrink-0 border-r border-white/5 bg-[#0B0D1E]">
          <DataSourcePanel onTableSelect={handleTableSelect} />
        </div>

        {/* Center Panel - Query Area */}
        <div className="flex-1 min-w-0 bg-[#0F1123]">
          <QueryPanel
            selectedTable={selectedTable}
            onQueryResult={handleQueryResult}
          />
        </div>

        {/* Right Panel - Results */}
        <div className="w-[460px] flex-shrink-0 border-l border-white/5 bg-[#0B0D1E]">
          <ResultsPanel queryResult={queryResult} />
        </div>
      </div>

      <Toaster
        theme="dark"
        position="top-right"
        toastOptions={{
          style: {
            background: '#1a1b3e',
            border: '1px solid rgba(92, 225, 230, 0.2)',
            color: '#e5e7eb',
          },
        }}
      />
    </div>
  );
}

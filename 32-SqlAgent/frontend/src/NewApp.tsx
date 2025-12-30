import React, { useState } from 'react';
import { FileUploadPanel } from './components/FileUploadPanel';
import { NewQueryPanel } from './components/NewQueryPanel';
import { ResultsPanel } from './components/ResultsPanel';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';

export default function NewApp() {
  const [selectedFile, setSelectedFile] = useState<any>(null);
  const [selectedFileId, setSelectedFileId] = useState<string>('');
  const [queryResult, setQueryResult] = useState<any>(null);

  const handleFileUploaded = (file: any) => {
    setSelectedFile(file);
    toast.success('文件上传成功', {
      description: `${file.name} 已成功上传`,
    });
  };

  const handleFileSelect = (fileId: string) => {
    setSelectedFileId(fileId);
  };

  const handleQueryResult = (result: any) => {
    setQueryResult(result);
    if (result.success) {
      toast.success('查询成功', {
        description: `返回 ${result.returnedRows || 0} 条结果`,
      });
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-gray-50">
      {/* Header */}
      <div className="h-16 border-b bg-white flex items-center px-6 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h1 className="text-lg font-semibold text-gray-900">
            SQL Agent
            <span className="text-gray-500 text-base ml-2">智能数据分析平台</span>
          </h1>
        </div>
        <div className="ml-auto flex items-center gap-2 text-sm text-gray-500">
          {selectedFile && (
            <>
              <span>当前文件：</span>
              <span className="font-medium text-gray-700">{selectedFile.name}</span>
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - File Upload */}
        <div className="w-[320px] flex-shrink-0 border-r bg-white">
          <FileUploadPanel
            onFileUploaded={handleFileUploaded}
            selectedFileId={selectedFileId}
            onFileSelect={handleFileSelect}
          />
        </div>

        {/* Center Panel - Query */}
        <div className="flex-1 min-w-0 bg-gray-50">
          <NewQueryPanel
            selectedFileId={selectedFileId}
            onQueryResult={handleQueryResult}
            onVisualizationRequest={(data) => console.log('Visualization requested:', data)}
          />
        </div>

        {/* Right Panel - Results */}
        <div className="w-[500px] flex-shrink-0 border-l bg-white">
          <ResultsPanel data={queryResult?.data || []} />
        </div>
      </div>

      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'white',
            border: '1px solid #e5e7eb',
            color: '#1f2937',
          },
        }}
      />
    </div>
  );
}
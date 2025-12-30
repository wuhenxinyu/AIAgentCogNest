import React, { useState, useEffect } from 'react';
import { Database, Table2, FileText, Upload, ChevronRight, Star } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { api } from '../services/api';

interface DataSource {
  name: string;
  table: string;
  rows: number;
  columns: string[];
  description: string;
  source: string;
  file_id?: string;
}

interface DataSourceManagerProps {
  selectedSource: DataSource | null;
  onSourceSelect: (source: DataSource) => void;
  onFileUploaded: (fileInfo: any) => void;
}

export function DataSourceManager({
  selectedSource,
  onSourceSelect,
  onFileUploaded
}: DataSourceManagerProps) {
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string[]>(['database', 'upload']);

  useEffect(() => {
    loadDataSources();
  }, []);

  const loadDataSources = async () => {
    try {
      setLoading(true);
      const response = await api.getDataSources();
      if (response.success) {
        // 分组数据源
        const databaseSources = response.sources.filter(s => s.source === 'mock' || s.source === 'derived');
        const realDataSources = response.sources.filter(s => s.source === 'csv');
        const uploadSources = response.sources.filter(s => s.source === 'upload');

        // 重新组织数据
        const organized: DataSource[] = [
          ...databaseSources,
          ...realDataSources,
          ...uploadSources
        ];

        setDataSources(organized);
      }
    } catch (error) {
      console.error('加载数据源失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append('file', file);

      // 这里简化处理，实际应该调用上传API
      onFileUploaded({
        filename: file.name,
        size: file.size,
        type: file.name.endsWith('.csv') ? 'csv' : 'excel'
      });

      // 重新加载数据源
      await loadDataSources();
    } catch (error) {
      console.error('文件上传失败:', error);
    }
  };

  const getSourceIcon = (source: DataSource) => {
    if (source.source === 'upload') {
      return <FileText className="w-4 h-4 text-green-500" />;
    } else if (source.source === 'csv' || source.source === 'real') {
      return <Star className="w-4 h-4 text-blue-500" />;
    } else {
      return <Database className="w-4 h-4 text-purple-500" />;
    }
  };

  const getSourceBadge = (source: DataSource) => {
    const variant = source.source === 'upload' ? 'secondary' :
                   source.source === 'csv' ? 'default' : 'outline';
    const text = source.source === 'upload' ? '上传文件' :
                  source.source === 'csv' ? '真实数据' :
                  source.source === 'mock' ? '模拟数据' : '汇总数据';

    return <Badge variant={variant} className="text-xs">{text}</Badge>;
  };

  const toggleExpand = (section: string) => {
    setExpanded(prev =>
      prev.includes(section)
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  };

  const groupedSources = {
    database: dataSources.filter(s => s.source === 'mock' || s.source === 'derived'),
    real: dataSources.filter(s => s.source === 'csv'),
    upload: dataSources.filter(s => s.source === 'upload')
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Database className="w-5 h-5" />
            数据源管理
          </CardTitle>
          <Badge variant="outline" className="text-xs">
            {dataSources.length} 个数据源
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <ScrollArea className="h-[600px]">
          <div className="space-y-4">
            {/* 模拟数据源 */}
            <div>
              <Button
                variant="ghost"
                className="w-full justify-between p-2"
                onClick={() => toggleExpand('database')}
              >
                <span className="flex items-center gap-2">
                  <Database className="w-4 h-4" />
                  ERP模拟数据
                </span>
                <ChevronRight className={`w-4 h-4 transition-transform ${
                  expanded.includes('database') ? 'rotate-90' : ''
                }`} />
              </Button>

              {expanded.includes('database') && (
                <div className="ml-4 mt-2 space-y-2">
                  {groupedSources.database.map((source) => (
                    <div
                      key={source.table}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedSource?.table === source.table
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => onSourceSelect(source)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            {getSourceIcon(source)}
                            <span className="font-medium text-sm truncate">
                              {source.name}
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 mb-2">
                            {source.description}
                          </p>
                          <div className="flex items-center gap-2">
                            {getSourceBadge(source)}
                            <span className="text-xs text-gray-400">
                              {source.rows.toLocaleString()} 行
                            </span>
                          </div>
                        </div>
                        {selectedSource?.table === source.table && (
                          <div className="w-2 h-2 bg-blue-500 rounded-full" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 真实数据源 */}
            <div>
              <Button
                variant="ghost"
                className="w-full justify-between p-2"
                onClick={() => toggleExpand('real')}
              >
                <span className="flex items-center gap-2">
                  <Star className="w-4 h-4 text-blue-500" />
                  真实数据
                </span>
                <ChevronRight className={`w-4 h-4 transition-transform ${
                  expanded.includes('real') ? 'rotate-90' : ''
                }`} />
              </Button>

              {expanded.includes('real') && (
                <div className="ml-4 mt-2 space-y-2">
                  {groupedSources.real.map((source) => (
                    <div
                      key={source.table}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedSource?.table === source.table
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => onSourceSelect(source)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            {getSourceIcon(source)}
                            <span className="font-medium text-sm truncate">
                              {source.name}
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 mb-2">
                            {source.description}
                          </p>
                          <div className="flex items-center gap-2">
                            {getSourceBadge(source)}
                            <span className="text-xs text-gray-400">
                              {source.rows.toLocaleString()} 行
                            </span>
                          </div>
                        </div>
                        {selectedSource?.table === source.table && (
                          <div className="w-2 h-2 bg-blue-500 rounded-full" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 上传文件 */}
            <div>
              <Button
                variant="ghost"
                className="w-full justify-between p-2"
                onClick={() => toggleExpand('upload')}
              >
                <span className="flex items-center gap-2">
                  <Upload className="w-4 h-4" />
                  文件上传
                </span>
                <ChevronRight className={`w-4 h-4 transition-transform ${
                  expanded.includes('upload') ? 'rotate-90' : ''
                }`} />
              </Button>

              {expanded.includes('upload') && (
                <div className="ml-4 mt-2 space-y-2">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center">
                    <input
                      type="file"
                      accept=".csv,.xlsx,.xls"
                      onChange={handleFileUpload}
                      className="hidden"
                      id="fileUpload"
                    />
                    <label htmlFor="fileUpload" className="cursor-pointer">
                      <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                      <p className="text-sm text-gray-600">
                        点击或拖拽CSV/Excel文件到此处
                      </p>
                    </label>
                  </div>

                  {groupedSources.upload.map((source) => (
                    <div
                      key={source.table}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedSource?.table === source.table
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => onSourceSelect(source)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            {getSourceIcon(source)}
                            <span className="font-medium text-sm truncate">
                              {source.name}
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 mb-2">
                            {source.description}
                          </p>
                          <div className="flex items-center gap-2">
                            {getSourceBadge(source)}
                            <span className="text-xs text-gray-400">
                              {source.rows.toLocaleString()} 行
                            </span>
                          </div>
                        </div>
                        {selectedSource?.table === source.table && (
                          <div className="w-2 h-2 bg-blue-500 rounded-full" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
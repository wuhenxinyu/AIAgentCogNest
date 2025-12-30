import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Database, Table2, Upload, Search, FileSpreadsheet } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { HoverCard, HoverCardContent, HoverCardTrigger } from './ui/hover-card';
import { api } from '../services/api';

interface TableNode {
  name: string;
  type: 'schema' | 'table' | 'column';
  children?: TableNode[];
  tableName?: string; // 实际的数据库表名
  metadata?: {
    fields?: number;
    rows?: number;
    sample?: string;
  };
}

interface DataSource {
  name: string;
  table: string;
  rows: number;
  columns: string[];
  description: string;
  source: string;
}

interface DataSourcePanelProps {
  onTableSelect?: (tableName: string) => void;
}

export function DataSourcePanel({ onTableSelect }: DataSourcePanelProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['数据库表', '上传的文件']));
  const [searchQuery, setSearchQuery] = useState('');
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedTable, setSelectedTable] = useState<string | null>(null);

  useEffect(() => {
    loadDataSources();
  }, []);

  const loadDataSources = async () => {
    try {
      setLoading(true);
      const response = await api.getDataSources();
      console.log('加载的数据源:', response);
      setDataSources(response.sources || []);
    } catch (error) {
      console.error('加载数据源失败:', error);
      setDataSources([]);
    } finally {
      setLoading(false);
    }
  };

  const toggleNode = (nodeName: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeName)) {
      newExpanded.delete(nodeName);
    } else {
      newExpanded.add(nodeName);
    }
    setExpandedNodes(newExpanded);
  };

  const convertToTableNodes = (sources: DataSource[]): TableNode[] => {
    // 按source分组
    const grouped = sources.reduce((acc, source) => {
      const category = source.source === 'upload' ? '上传的文件' : '数据库表';
      if (!acc[category]) {
        acc[category] = [];
      }
      acc[category].push(source);
      return acc;
    }, {} as Record<string, DataSource[]>);

    return Object.entries(grouped).map(([category, sources]) => ({
      name: category,
      type: 'schema' as const,
      children: sources.map(source => ({
        name: source.name,
        type: 'table' as const,
        tableName: source.table, // 实际的数据库表名
        metadata: {
          fields: Array.isArray(source.columns) ? source.columns.length : 0,
          rows: source.rows || 0,
          sample: Array.isArray(source.columns) 
            ? source.columns.slice(0, 3).join(', ') + (source.columns.length > 3 ? '...' : '')
            : ''
        },
        children: Array.isArray(source.columns) ? source.columns.map(column => ({
          name: column,
          type: 'column' as const
        })) : []
      }))
    }));
  };

  const filteredDataSources = convertToTableNodes(dataSources).filter(ds =>
    ds.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    ds.children?.some(table => table.name.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const renderTreeNode = (node: TableNode, level: number = 0) => {
    const isExpanded = expandedNodes.has(node.name);
    const hasChildren = node.children && node.children.length > 0;

    const icon = node.type === 'schema' ? (
      <Database className="w-3.5 h-3.5 text-cyan-400" />
    ) : node.type === 'table' ? (
      <Table2 className="w-3.5 h-3.5 text-blue-400" />
    ) : (
      <div className="w-1.5 h-1.5 rounded-full bg-purple-400/60 ml-1" />
    );

    const handleClick = () => {
      console.log('点击节点:', node.name, '类型:', node.type, '有子节点:', hasChildren);

      // 如果是table节点，优先选择表格
      if (node.type === 'table') {
        setSelectedTable(node.name);
        console.log('选中表格:', node.name);

        if (onTableSelect) {
          // 使用实际的数据库表名
          const actualTableName = node.tableName || node.name;
          console.log('传递表名:', actualTableName);
          onTableSelect(actualTableName);
        }

        // 如果table有子节点（字段），也展开/折叠
        if (hasChildren) {
          toggleNode(node.name);
        }
      } else if (hasChildren) {
        // schema节点只展开/折叠
        toggleNode(node.name);
      }
    };

    const content = (
      <div
        key={node.name}
        style={{ paddingLeft: `${level * 12}px` }}
        className="group"
      >
        <div
          className={`flex items-center gap-2 px-2 py-1.5 hover:bg-white/5 cursor-pointer rounded-md transition-colors ${
            node.type === 'table' && selectedTable === node.name ? 'bg-cyan-500/10 border-l-2 border-cyan-500' : ''
          }`}
          onClick={handleClick}
        >
          {hasChildren && (
            isExpanded ?
              <ChevronDown className="w-3 h-3 text-gray-500 flex-shrink-0" /> :
              <ChevronRight className="w-3 h-3 text-gray-500 flex-shrink-0" />
          )}
          {!hasChildren && <div className="w-3 flex-shrink-0" />}
          {icon}
          <span className="text-gray-300 text-xs truncate">{node.name}</span>
        </div>
      </div>
    );

    if (node.type === 'table' && node.metadata) {
      return (
        <React.Fragment key={node.name}>
          <HoverCard openDelay={300}>
            <HoverCardTrigger asChild>
              {content}
            </HoverCardTrigger>
            <HoverCardContent side="right" className="w-80 bg-[#1a1b3e] border-cyan-500/30 text-gray-200">
              <div className="space-y-2">
                <p className="text-xs text-cyan-400">表详情</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-400">字段数:</span>
                    <span className="ml-2 text-white">{node.metadata.fields}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">行数:</span>
                    <span className="ml-2 text-white">{node.metadata.rows?.toLocaleString()}</span>
                  </div>
                </div>
                <div>
                  <p className="text-gray-400 text-xs">示例字段:</p>
                  <p className="text-purple-300 text-xs font-mono mt-1">{node.metadata.sample}</p>
                </div>
              </div>
            </HoverCardContent>
          </HoverCard>
          {isExpanded && node.children?.map(child => renderTreeNode(child, level + 1))}
        </React.Fragment>
      );
    }

    return (
      <React.Fragment key={node.name}>
        {content}
        {isExpanded && node.children?.map(child => renderTreeNode(child, level + 1))}
      </React.Fragment>
    );
  };

  if (loading) {
    return (
      <div className="h-full flex flex-col bg-[#0d0e23] overflow-hidden">
        <div className="px-6 py-3 border-b border-white/10">
          <h2 className="text-cyan-400">数据源</h2>
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
      <div className="px-4 py-4 border-b border-white/5 flex-shrink-0">
        <h2 className="text-cyan-400 font-medium text-sm">数据源与文件</h2>
      </div>

      {/* Search */}
      <div className="px-3 py-3 border-b border-white/5">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
          <Input
            placeholder="搜索表或段..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-9 bg-[#13152E] border-white/5 text-gray-300 placeholder-gray-600 text-sm rounded-lg focus:border-cyan-500/30"
          />
        </div>
      </div>

      {/* Data Sources List */}
      <div className="flex-1 overflow-y-auto min-h-0 px-2 py-2">
        {/* Table Tree */}
        <div className="space-y-1">
          {filteredDataSources.length > 0 ? (
            filteredDataSources.map(ds => renderTreeNode(ds, 0))
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-500 text-xs">暂无数据源</p>
              <p className="text-gray-600 text-xs mt-1">请上传文件或连接数据库</p>
            </div>
          )}
        </div>

        {/* Upload Section */}
        <div className="mt-4 pt-4 border-t border-white/5">
          <div className="px-2">
            <p className="text-xs text-gray-500 mb-2 font-medium">已上传文件</p>
            <div className="space-y-1 mb-2">
              {dataSources.filter(ds => ds.source === 'upload').map((file, idx) => {
                // 使用 table 字段（file_{file_id}）作为选中标识
                const tableName = file.table || `file_${(file as any).file_id}` || file.name;
                const isSelected = selectedTable === tableName;
                
                return (
                  <div 
                    key={idx} 
                    onClick={() => {
                      if (onTableSelect) {
                        onTableSelect(tableName);
                        setSelectedTable(tableName);
                      }
                    }}
                    className={`flex items-center gap-2 px-3 py-2 bg-[#13152E] rounded-lg border border-white/5 cursor-pointer hover:border-cyan-500/20 hover:bg-cyan-500/5 transition-colors ${
                      isSelected ? 'border-cyan-500/40 bg-cyan-500/10' : ''
                    }`}
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                    <span className="text-xs text-gray-300 flex-1 truncate">{file.name}</span>
                  </div>
                );
              })}
            </div>
            <Button
              variant="outline"
              className="w-full h-9 bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 hover:text-gray-900 text-xs font-normal"
              onClick={() => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.csv,.xlsx,.xls';
                input.onchange = async (e: any) => {
                  const file = e.target.files[0];
                  if (file) {
                    try {
                      await api.uploadFile(file);
                      loadDataSources();
                    } catch (error) {
                      console.error('上传失败:', error);
                      alert('上传失败: ' + error);
                    }
                  }
                };
                input.click();
              }}
            >
              <Upload className="w-3.5 h-3.5 mr-2 text-gray-700" />
              上传文件 (CSV/Excel)
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
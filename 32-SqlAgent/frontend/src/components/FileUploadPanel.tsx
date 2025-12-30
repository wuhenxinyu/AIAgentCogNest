import React, { useState, useRef } from 'react';
import { Upload, FileText, Table2, X, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { uploadFileHelper } from '../services/api';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  columns: string[];
  totalColumns: number;
  estimatedRows: number;
  uploadTime: Date;
}

interface FileUploadPanelProps {
  onFileUploaded: (file: UploadedFile) => void;
  selectedFileId?: string;
  onFileSelect: (fileId: string) => void;
}

export function FileUploadPanel({
  onFileUploaded,
  selectedFileId,
  onFileSelect
}: FileUploadPanelProps) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // 检查文件类型
    const validTypes = ['text/csv', 'application/vnd.ms-excel',
                      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    if (!validTypes.includes(file.type) &&
        !file.name.endsWith('.csv') &&
        !file.name.endsWith('.xlsx') &&
        !file.name.endsWith('.xls')) {
      setUploadError('请上传CSV或Excel文件');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setUploadError(null);

    // 模拟上传进度
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      const result = await uploadFileHelper(file);
      clearInterval(progressInterval);
      setUploadProgress(100);

      if (result.success) {
        const newFile: UploadedFile = {
          id: result.fileId,
          name: result.filename,
          size: file.size,
          columns: result.headers,
          totalColumns: result.totalColumns,
          estimatedRows: result.estimatedRows,
          uploadTime: new Date()
        };

        setUploadedFiles(prev => [...prev, newFile]);
        onFileUploaded(newFile);
      } else {
        setUploadError(result.error || '上传失败');
      }
    } catch (error: any) {
      clearInterval(progressInterval);
      setUploadError(error.message || '上传失败');
    } finally {
      setIsUploading(false);
      setTimeout(() => {
        setUploadProgress(0);
      }, 1000);
    }
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
    if (selectedFileId === fileId) {
      onFileSelect('');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <FileText className="w-5 h-5" />
          文件管理
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 上传区域 */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isUploading}
          />

          <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />

          {!isUploading && (
            <div>
              <p className="text-sm text-gray-600 mb-2">
                拖拽CSV或Excel文件到此处，或点击选择文件
              </p>
              <Button
                size="sm"
                onClick={() => fileInputRef.current?.click()}
              >
                选择文件
              </Button>
            </div>
          )}

          {isUploading && (
            <div className="space-y-2">
              <p className="text-sm text-gray-600">正在上传...</p>
              <Progress value={uploadProgress} className="w-full" />
              <p className="text-xs text-gray-500">{uploadProgress}%</p>
            </div>
          )}

          {uploadError && (
            <div className="flex items-center gap-2 text-red-500 text-sm mt-2">
              <AlertCircle className="w-4 h-4" />
              <span>{uploadError}</span>
            </div>
          )}
        </div>

        {/* 已上传文件列表 */}
        <div className="space-y-2">
          <h3 className="text-sm font-medium">已上传文件</h3>
          {uploadedFiles.length === 0 ? (
            <p className="text-sm text-gray-500 text-center py-4">
              暂无上传文件
            </p>
          ) : (
            uploadedFiles.map((file) => (
              <div
                key={file.id}
                className={`border rounded-lg p-3 cursor-pointer transition-colors ${
                  selectedFileId === file.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => onFileSelect(file.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-2 flex-1 min-w-0">
                    <Table2 className="w-4 h-4 mt-0.5 text-gray-400 flex-shrink-0" />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium truncate">{file.name}</p>
                      <p className="text-xs text-gray-500">
                        {formatFileSize(file.size)} • {file.totalColumns}列 •
                        约{file.estimatedRows.toLocaleString()}行
                      </p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {file.columns.slice(0, 3).map((col, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            {col}
                          </Badge>
                        ))}
                        {file.columns.length > 3 && (
                          <Badge variant="outline" className="text-xs">
                            +{file.columns.length - 3}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1">
                    {selectedFileId === file.id && (
                      <CheckCircle className="w-4 h-4 text-blue-500" />
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(file.id);
                      }}
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
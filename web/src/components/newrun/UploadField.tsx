interface Props {
  file: File | null;
  error?: string;
  onChange: (file: File | null) => void;
}

export function UploadField({ file, error, onChange }: Props) {
  return (
    <div className="mt-6">
      <label className="block text-sm text-slate-300">
        上传包
        <input
          type="file"
          accept=".zip,.tar,.tar.gz,.tgz"
          onChange={(event) => onChange(event.target.files?.[0] ?? null)}
          className="mt-2 block w-full cursor-pointer rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-emerald-500 file:px-3 file:py-1.5 file:text-slate-950"
        />
      </label>
      <p className="mt-1 text-xs text-slate-500">
        支持 zip 或 tar，后端按文件魔数识别；解压需要 sandbox-runner 运行中。
      </p>
      {file && (
        <p className="mt-1 text-xs text-slate-400">
          已选择：{file.name}（{formatSize(file.size)}）
        </p>
      )}
      {error && <p className="mt-1 text-xs text-rose-400">{error}</p>}
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

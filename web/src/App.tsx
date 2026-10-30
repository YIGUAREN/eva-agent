import { Navigate, Route, Routes } from "react-router-dom";

import { useSession } from "./hooks/useSession";

export function App() {
  const { session, loading } = useSession();

  if (loading) {
    return <main className="grid min-h-screen place-items-center text-slate-400">加载中…</main>;
  }
  if (!session) {
    return <main className="p-10">请先登录（Task 14 添加登录页）。</main>;
  }
  return (
    <Routes>
      <Route path="/" element={<main className="p-10">Eva Agent</main>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

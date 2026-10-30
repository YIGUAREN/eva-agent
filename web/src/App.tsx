import { Navigate, Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { useSession } from "./hooks/useSession";
import { LoginPage } from "./pages/LoginPage";
import { NewRunPage } from "./pages/NewRunPage";
import { PublicRunPage } from "./pages/PublicRunPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { RunsPage } from "./pages/RunsPage";

export function App() {
  const { session, loading, refresh } = useSession();

  if (loading) {
    return <main className="grid min-h-screen place-items-center text-slate-400">加载中…</main>;
  }

  if (!session) {
    return (
      <Routes>
        <Route path="/share/:runId" element={<PublicRunPage />} />
        <Route path="*" element={<LoginPage onLogin={refresh} />} />
      </Routes>
    );
  }

  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<RunsPage />} />
        <Route path="/new" element={<NewRunPage />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
        <Route path="/share/:runId" element={<PublicRunPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

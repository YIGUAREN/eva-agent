import { NavLink, Outlet } from "react-router-dom";

import { api } from "../api/client";

const link = "rounded-lg px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800 hover:text-white";
const active = "bg-slate-800 text-white";

export function Layout() {
  async function logout() {
    try {
      await api.logout();
    } finally {
      window.location.assign("/");
    }
  }

  return (
    <div className="min-h-screen bg-slate-950">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-6">
            <NavLink to="/" className="flex items-center gap-2 text-lg font-semibold">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-emerald-500 text-sm font-bold text-slate-950">E</span>
              Eva Agent
            </NavLink>
            <nav className="flex items-center gap-1">
              <NavLink to="/" end className={({ isActive }) => `${link} ${isActive ? active : ""}`}>运行</NavLink>
              <NavLink to="/new" className={({ isActive }) => `${link} ${isActive ? active : ""}`}>新建评估</NavLink>
            </nav>
          </div>
          <button onClick={logout} className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:border-slate-500">
            退出
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}

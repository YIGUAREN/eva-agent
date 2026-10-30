import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
        // 浏览器开发页面的 Origin 是 http://localhost:5173；登录端点要求 Origin
        // 与 public_base_url（文档中的 http://127.0.0.1:8000）一致，因此代理把
        // Origin 改写为该值，使 `pnpm dev` 下的登录可以通过。
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            proxyReq.setHeader("origin", "http://127.0.0.1:8000");
          });
        },
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});

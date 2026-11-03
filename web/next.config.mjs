import path from "node:path";
import { fileURLToPath } from "node:url";

const dir = path.dirname(fileURLToPath(import.meta.url));
const ui = path.resolve(dir, "ui");

/**
 * Landing page served by bot.py at `/`.
 *
 * `output: "export"` writes plain HTML/CSS/JS to `web/out/`, so the Python
 * server can serve it without Node running. Rebuild with `npm run build`.
 */
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "export",
  images: { unoptimized: true },

  // The UI kit lives in ./ui as TypeScript source; alias it so components can
  // keep importing "@receptionai/ui". tsconfig.json has matching `paths`.
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "@receptionai/ui/tailwind-preset": path.join(ui, "tailwind-preset.ts"),
      "@receptionai/ui/styles": path.join(ui, "src/styles"),
      "@receptionai/ui": path.join(ui, "src/index.ts"),
    };
    return config;
  },
};

export default nextConfig;

import type { Config } from "tailwindcss";
import preset from "./ui/tailwind-preset";

export default {
  presets: [preset],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    // The UI kit's classes must be scanned too, or its utilities get purged.
    "./ui/src/**/*.{ts,tsx}",
  ],
} satisfies Config;

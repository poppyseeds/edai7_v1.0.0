/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0a0d12",
        panel: "#111720",
        line: "#273142",
        ink: "#e7edf7",
        muted: "#94a3b8",
        signal: "#35d4a3",
        electric: "#66a6ff",
      },
      boxShadow: { panel: "0 18px 48px rgba(0, 0, 0, 0.22)" },
    },
  },
  plugins: [],
};

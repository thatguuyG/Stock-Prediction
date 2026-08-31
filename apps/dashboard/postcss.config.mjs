// Tailwind v4 ships its PostCSS integration as a separate package, and the
// plugin now handles vendor prefixing itself — no autoprefixer entry needed.
const config = {
  plugins: {
    '@tailwindcss/postcss': {},
  },
};

export default config;

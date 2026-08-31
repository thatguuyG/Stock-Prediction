// Flat config. Next 16 removed `next lint`, so linting runs through the eslint
// CLI directly (see the `lint` target in project.json).
import nextCoreWebVitals from 'eslint-config-next/core-web-vitals';
import nextTypeScript from 'eslint-config-next/typescript';

const config = [
  { ignores: ['.next/**', 'next-env.d.ts'] },
  ...nextCoreWebVitals,
  ...nextTypeScript,
];

export default config;

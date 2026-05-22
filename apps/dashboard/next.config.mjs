/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy /api/* to the FastAPI shim in dev. Production routing is decided in Phase 4.
  async rewrites() {
    const apiHost = process.env.API_HOST || '127.0.0.1';
    const apiPort = process.env.API_PORT || '8000';
    return [
      {
        source: '/api/:path*',
        destination: `http://${apiHost}:${apiPort}/:path*`,
      },
    ];
  },
};

export default nextConfig;

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@radix-ui/react-icons"],
  experimental: {
    esmExternals: true,
  },
};

module.exports = nextConfig;

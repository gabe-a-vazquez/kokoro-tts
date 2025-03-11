/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  transpilePackages: ["@radix-ui/react-icons"],
};

module.exports = nextConfig;

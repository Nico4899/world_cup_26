import type { NextConfig } from "next";
import createMDX from "@next/mdx";

// Turbopack requires plugin specifiers be serialisable (strings), not
// imported function references. `remark-gfm` resolves via Node's module
// resolution at compile time.
const withMDX = createMDX({
  options: {
    remarkPlugins: [["remark-gfm", {}]],
    rehypePlugins: [],
  },
});

const nextConfig: NextConfig = {
  // Allow .md / .mdx files to be importable as React components, so the
  // About page can `import Methodology from "@/content/methodology.mdx"`
  // and Next compiles it via @next/mdx at build time.
  pageExtensions: ["ts", "tsx", "md", "mdx"],
};

export default withMDX(nextConfig);

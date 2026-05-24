#!/usr/bin/env node
// Sync docs/methodology.md → frontend/src/content/methodology.mdx.
//
// Why this exists: the About page imports the methodology as MDX so Next can
// compile it once at build time. We don't want two copies of the same text in
// git (the .md is the authoritative source for the README + ARCHITECTURE
// docs), so this script copies the .md verbatim into the frontend's content/
// directory. Wired into `pnpm prebuild` + `pnpm predev` so the .mdx file is
// always fresh; check the .mdx into git too so Vercel builds work without
// running the script first.

import { copyFileSync, mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..", "..", "docs", "methodology.md");
const DST = resolve(HERE, "..", "src", "content", "methodology.mdx");

if (!existsSync(SRC)) {
  console.error(`sync-methodology: source missing at ${SRC}`);
  process.exit(1);
}

mkdirSync(dirname(DST), { recursive: true });

// MDX is strict about JSX-like syntax. The methodology has `<` characters in
// inequalities (e.g. "≥250", math like "λ_home"). Wrapping the body in a
// {/* prose */} comment is unnecessary; we just escape `<` and `{` that look
// JSX-y. For now the docs/methodology.md content is JSX-safe, so a verbatim
// copy works. Add escaping logic here only if a future commit breaks the build.
copyFileSync(SRC, DST);

// Drop a leading note so it's clear the .mdx is generated.
const body = readFileSync(DST, "utf8");
const header = "{/* Generated from docs/methodology.md by scripts/sync-methodology.mjs — do not edit directly. */}\n\n";
if (!body.startsWith("{/* Generated")) {
  writeFileSync(DST, header + body);
}

console.log(`sync-methodology: ${SRC} → ${DST}`);

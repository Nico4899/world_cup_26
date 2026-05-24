import type { MDXComponents } from "mdx/types";
import type { ComponentPropsWithoutRef, ReactNode } from "react";

/**
 * Type-safe Tailwind defaults for `next-mdx-remote` / `@next/mdx`'s
 * rendered nodes. Without this Next would emit raw `<h1>` / `<p>` without
 * the prose styling that the About page expects.
 *
 * Heading nodes also get auto-slugged `id` attributes so the in-page ToC
 * sidebar (and any `glossary.link?` entry) can anchor-jump to them.
 */
export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h1: ({ children, ...props }) => (
      <h1 id={slugify(children)} className="text-2xl font-semibold mt-4 mb-2 scroll-mt-20" {...props}>
        {children}
      </h1>
    ),
    h2: ({ children, ...props }) => (
      <h2 id={slugify(children)} className="text-xl font-semibold mt-6 mb-2 scroll-mt-20" {...props}>
        {children}
      </h2>
    ),
    h3: ({ children, ...props }) => (
      <h3 id={slugify(children)} className="text-lg font-semibold mt-4 mb-2 scroll-mt-20" {...props}>
        {children}
      </h3>
    ),
    p: (props) => <p className="text-sm leading-6 my-2" {...props} />,
    ul: (props) => <ul className="text-sm list-disc pl-6 my-2 space-y-1" {...props} />,
    ol: (props) => <ol className="text-sm list-decimal pl-6 my-2 space-y-1" {...props} />,
    li: (props) => <li className="leading-6" {...props} />,
    code: (props: ComponentPropsWithoutRef<"code">) => (
      <code className="rounded bg-muted px-1 py-0.5 text-xs" {...props} />
    ),
    pre: (props) => (
      <pre className="rounded-md border bg-muted/30 p-3 text-xs overflow-x-auto my-3" {...props} />
    ),
    table: (props) => (
      <div className="my-3 overflow-x-auto">
        <table className="text-sm border-collapse" {...props} />
      </div>
    ),
    th: (props) => (
      <th className="border-b font-medium px-2 py-1 text-left" {...props} />
    ),
    td: (props) => <td className="border-b px-2 py-1" {...props} />,
    a: (props) => (
      <a className="text-primary underline-offset-2 hover:underline" {...props} />
    ),
    ...components,
  };
}

/**
 * Reduce MDX children (which may be strings, arrays, or React nodes) down
 * to a plain-text representation, then slugify for `id=`. Handles the
 * three shapes MDX emits: string, array, and `{ props: { children } }`.
 */
function slugify(node: ReactNode): string {
  return collectText(node)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function collectText(node: ReactNode): string {
  if (node == null || node === false || node === true) return "";
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(collectText).join("");
  if (typeof node === "object" && "props" in node) {
    const props = (node as { props?: { children?: ReactNode } }).props;
    return collectText(props?.children);
  }
  return "";
}

import type { MDXComponents } from "mdx/types";
import type { ComponentPropsWithoutRef } from "react";

/**
 * Type-safe Tailwind defaults for `next-mdx-remote` / `@next/mdx`'s
 * rendered nodes. Without this Next would emit raw `<h1>` / `<p>` without
 * the prose styling that the About page expects.
 */
export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h1: (props) => <h1 className="text-2xl font-semibold mt-4 mb-2" {...props} />,
    h2: (props) => <h2 className="text-xl font-semibold mt-6 mb-2" {...props} />,
    h3: (props) => <h3 className="text-lg font-semibold mt-4 mb-2" {...props} />,
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

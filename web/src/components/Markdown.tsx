import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";

/** Shared markdown renderer: GitHub tables, LaTeX math, heading anchors.
 *  `urlTransform` can be overridden (e.g. to keep the internal `cite:` scheme
 *  that react-markdown's default sanitizer would otherwise strip). */
export default function Markdown({
  children,
  components,
  urlTransform,
}: {
  children: string;
  components?: Components;
  urlTransform?: (url: string) => string;
}) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeSlug]}
        components={components}
        urlTransform={urlTransform}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

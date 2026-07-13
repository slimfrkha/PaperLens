import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
  componentStack: string | null;
}

/**
 * Catches render/effect errors that would otherwise unmount the whole tree and
 * leave a blank page. Renders the error and a reload button instead, using plain
 * HTML so it works even when the failure is in the component library.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, componentStack: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Uncaught render error:", error, info.componentStack);
    this.setState({ componentStack: info.componentStack ?? null });
  }

  render() {
    const { error, componentStack } = this.state;
    if (!error) return this.props.children;
    return (
      <div style={{ padding: 24, fontFamily: "system-ui, sans-serif", maxWidth: 720 }}>
        <h2 style={{ marginTop: 0 }}>Something broke on this page</h2>
        <p>The view crashed instead of rendering. Reload to recover.</p>
        <pre
          style={{
            whiteSpace: "pre-wrap",
            background: "rgba(127,127,127,0.12)",
            padding: 12,
            borderRadius: 6,
            overflowX: "auto",
          }}
        >
          {error.message}
          {error.stack ? `\n\n${error.stack}` : ""}
          {componentStack ? `\n\nComponent stack:${componentStack}` : ""}
        </pre>
        <button
          onClick={() => window.location.reload()}
          style={{ padding: "8px 16px", cursor: "pointer" }}
        >
          Reload
        </button>
      </div>
    );
  }
}

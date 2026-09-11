import { useCallback, useMemo } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import PaperPanel from "../components/PaperPanel";

/** Standalone paper route (`/papers/:id`). The paper reader itself lives in `PaperPanel`
 *  (shared with the chat side-by-side view); this wrapper just maps the route param and the
 *  one-time citation-highlight carried in router state onto its props. */
export default function PaperViewer() {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation() as {
    key: string;
    pathname: string;
    state?: { highlight?: string; section?: string };
  };
  const snippet = location.state?.highlight;
  const section = location.state?.section;
  // Stable object (PaperPanel's highlight effect keys on identity): a new object only when
  // the cited passage changes, so unrelated re-renders don't re-trigger the jump.
  const highlight = useMemo(() => (snippet ? { snippet, section } : null), [snippet, section]);
  const pathname = location.pathname;
  // Stable identity so PaperPanel's memo holds across unrelated re-renders.
  const onHighlightConsumed = useCallback(
    () => navigate(pathname, { replace: true, state: {} }),
    [navigate, pathname],
  );
  if (!id) return null;
  return (
    <PaperPanel
      paperId={id}
      variant="page"
      highlight={highlight}
      onHighlightConsumed={onHighlightConsumed}
    />
  );
}

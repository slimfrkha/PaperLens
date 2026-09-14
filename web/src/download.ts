// The slice of the File System Access API we use — narrowly typed so we don't pull in `any`.
// Present on Chromium browsers, absent on Firefox/Safari (we fall back there).
type SaveFilePicker = (opts?: {
  suggestedName?: string;
  types?: { description?: string; accept: Record<string, string[]> }[];
}) => Promise<{
  createWritable: () => Promise<{
    write: (data: string) => Promise<void>;
    close: () => Promise<void>;
  }>;
}>;

/** Saves `text` as a file named `filename`. Where the API exists (Chromium), opens a native
 *  Save dialog so the reader chooses the folder and can edit the name; otherwise falls back
 *  to a regular download (suggested name → the browser's Downloads folder). Resolves once the
 *  file is written or the reader cancels. */
export async function downloadTextFile(
  filename: string,
  text: string,
  mime = "text/markdown",
): Promise<void> {
  const picker = (window as unknown as { showSaveFilePicker?: SaveFilePicker }).showSaveFilePicker;
  if (typeof picker === "function") {
    let handle: Awaited<ReturnType<SaveFilePicker>> | undefined;
    try {
      handle = await picker({
        suggestedName: filename,
        types: [{ description: "Markdown", accept: { "text/markdown": [".md"] } }],
      });
    } catch (e) {
      // Reader dismissed the dialog — nothing to save.
      if (e instanceof DOMException && e.name === "AbortError") return;
      // The dialog couldn't open (e.g. not a user gesture): fall back to the anchor download.
    }
    // Once a file is chosen, commit to it — a write failure surfaces rather than silently
    // re-downloading to Downloads (which would leave two files).
    if (handle) {
      const writable = await handle.createWritable();
      await writable.write(text);
      await writable.close();
      return;
    }
  }
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** A filename-safe slug of a chat name: lowercase, non-alphanumerics collapsed to single
 *  dashes, trimmed, capped to a sane length. Falls back to `fallback` (itself slugified)
 *  when the name has no usable characters left (e.g. a CJK/accented title under this
 *  ASCII-only rule), and to `"chat"` if that is empty too. */
export function slugFilename(name: string, fallback: string): string {
  const slugify = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 80)
      .replace(/-+$/, "");
  return slugify(name) || slugify(fallback) || "chat";
}

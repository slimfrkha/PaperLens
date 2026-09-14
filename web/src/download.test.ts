import { afterEach, describe, expect, it, vi } from "vitest";
import { downloadTextFile, slugFilename } from "./download";

function stubPicker(impl: unknown) {
  (window as unknown as { showSaveFilePicker?: unknown }).showSaveFilePicker = impl;
}

describe("downloadTextFile", () => {
  afterEach(() => {
    delete (window as unknown as { showSaveFilePicker?: unknown }).showSaveFilePicker;
  });

  it("uses the native save picker (with the suggested name) and writes the content", async () => {
    const written: string[] = [];
    const writable = {
      write: (d: string) => {
        written.push(d);
        return Promise.resolve();
      },
      close: () => Promise.resolve(),
    };
    const picker = vi.fn().mockResolvedValue({ createWritable: () => Promise.resolve(writable) });
    stubPicker(picker);

    await downloadTextFile("my-chat.md", "hello");

    expect(picker).toHaveBeenCalledWith(expect.objectContaining({ suggestedName: "my-chat.md" }));
    expect(written.join("")).toBe("hello");
  });

  it("returns quietly when the reader cancels the save dialog", async () => {
    stubPicker(vi.fn().mockRejectedValue(new DOMException("cancelled", "AbortError")));
    await expect(downloadTextFile("x.md", "y")).resolves.toBeUndefined();
  });

  it("falls back to an anchor download when no save picker is available", async () => {
    // afterEach removes any picker; stub the Blob-URL API jsdom doesn't implement.
    const createURL = vi.fn().mockReturnValue("blob:x");
    const revokeURL = vi.fn();
    const urlApi = URL as unknown as { createObjectURL: unknown; revokeObjectURL: unknown };
    urlApi.createObjectURL = createURL;
    urlApi.revokeObjectURL = revokeURL;
    const clicked: HTMLAnchorElement[] = [];
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      clicked.push(this);
    });
    try {
      await downloadTextFile("out.md", "body");
      expect(createURL).toHaveBeenCalled();
      expect(clicked).toHaveLength(1);
      expect(clicked[0].download).toBe("out.md");
      expect(revokeURL).toHaveBeenCalledWith("blob:x");
    } finally {
      clickSpy.mockRestore();
      delete (URL as unknown as { createObjectURL?: unknown }).createObjectURL;
      delete (URL as unknown as { revokeObjectURL?: unknown }).revokeObjectURL;
    }
  });
});

describe("slugFilename", () => {
  it("slugifies a normal name", () => {
    expect(slugFilename("Transformers vs Mamba!", "fallback")).toBe("transformers-vs-mamba");
  });

  it("falls back (slugified) when the name has no usable characters", () => {
    expect(slugFilename("日本語", "chat-1c90796019f5")).toBe("chat-1c90796019f5");
  });

  it("falls back to 'chat' when both name and fallback slug to empty", () => {
    expect(slugFilename("！！！", "…")).toBe("chat");
  });

  it("caps length and never leaves a trailing dash", () => {
    const slug = slugFilename("a ".repeat(100), "fallback");
    expect(slug.length).toBeLessThanOrEqual(80);
    expect(slug.endsWith("-")).toBe(false);
  });
});

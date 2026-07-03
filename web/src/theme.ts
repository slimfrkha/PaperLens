import { createTheme } from "@mantine/core";
import { palettes } from "./palettes";

// ── Active palette ── Rosé Pine (Dawn / Main). Alternatives remain defined in
//    palettes.ts: palettes.solarized | palettes.catppuccin | palettes.gruvbox
const ACTIVE = palettes.rosePine;

export const theme = createTheme({
  primaryColor: "accent",
  primaryShade: ACTIVE.primaryShade,
  colors: { accent: ACTIVE.accent, gray: ACTIVE.gray, dark: ACTIVE.dark },
  white: ACTIVE.white,
  black: ACTIVE.black,

  fontFamily:
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  fontFamilyMonospace:
    "'SFMono-Regular', 'SF Mono', ui-monospace, 'JetBrains Mono', Menlo, monospace",

  // Serif (Newsreader) for titles and reading surfaces — the "publication" voice.
  headings: {
    fontFamily: "'Newsreader', Georgia, 'Times New Roman', serif",
    fontWeight: "500",
    sizes: {
      h1: { fontSize: "2rem", lineHeight: "1.15" },
      h2: { fontSize: "1.55rem", lineHeight: "1.2" },
      h3: { fontSize: "1.25rem", lineHeight: "1.3" },
      h4: { fontSize: "1.05rem", lineHeight: "1.4" },
    },
  },

  defaultRadius: "md",
  radius: { xs: "4px", sm: "6px", md: "10px", lg: "14px", xl: "20px" },

  shadows: {
    xs: "0 1px 2px rgba(31,28,23,0.04)",
    sm: "0 1px 2px rgba(31,28,23,0.04), 0 2px 6px rgba(31,28,23,0.05)",
    md: "0 4px 16px rgba(31,28,23,0.07)",
    lg: "0 12px 32px rgba(31,28,23,0.10)",
  },

  components: {
    Button: {
      defaultProps: { fw: 500 },
      styles: { root: { transition: "background-color 120ms ease, transform 120ms ease" } },
    },
    Anchor: { defaultProps: { underline: "never" } },
    Tooltip: { defaultProps: { radius: "sm", withArrow: true, openDelay: 200 } },
    // Long hyphenated tags read calmer in normal case than shouty uppercase.
    Badge: { styles: { root: { textTransform: "none", fontWeight: 500 } } },
  },
});

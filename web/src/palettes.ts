import type { MantineColorShade, MantineColorsTuple } from "@mantine/core";

/** A famous public light+dark palette mapped onto Mantine's scales.
 *  - `gray`  is the LIGHT-mode neutral ramp (0 = lightest surface … 9 = darkest text)
 *  - `dark`  is the DARK-mode ramp (0 = lightest text … 7 = body background … 9 = darkest)
 *  - `accent` is the interactive/citation hue (10-step ramp)
 *  Custom surfaces/borders/text read scheme-aware tokens (see styles.css), so
 *  setting these five scales is all a palette needs to look right in both modes. */
export interface Palette {
  name: string;
  white: string;
  black: string;
  accent: MantineColorsTuple;
  gray: MantineColorsTuple;
  dark: MantineColorsTuple;
  primaryShade: { light: MantineColorShade; dark: MantineColorShade };
}

// ── Solarized ── Ethan Schoonover's canonical warm light + teal-ink dark.
export const solarized: Palette = {
  name: "Solarized",
  white: "#fdf6e3",
  black: "#586e75",
  accent: [
    "#e8f1fb",
    "#cfe4f6",
    "#a6cded",
    "#77b3e3",
    "#4f9dd9",
    "#268bd2",
    "#2380c4",
    "#1f72af",
    "#1a5f92",
    "#154d78",
  ],
  gray: [
    "#fdf6e3",
    "#f4ecd8",
    "#eee8d5",
    "#ded8c4",
    "#c4c9bf",
    "#93a1a1",
    "#839496",
    "#657b83",
    "#586e75",
    "#073642",
  ],
  dark: [
    "#93a1a1",
    "#839496",
    "#657b83",
    "#586e75",
    "#274550",
    "#0d3b47",
    "#073642",
    "#002b36",
    "#00232c",
    "#001b22",
  ],
  primaryShade: { light: 6, dark: 4 },
};

// ── Catppuccin ── Latte (light) + Mocha (dark), soft modern pastels.
export const catppuccin: Palette = {
  name: "Catppuccin",
  white: "#eff1f5",
  black: "#4c4f69",
  accent: [
    "#e7edfe",
    "#cdd9fd",
    "#a6bffb",
    "#79a0f9",
    "#4a83f7",
    "#1e66f5",
    "#195ce0",
    "#164ec0",
    "#1341a0",
    "#0f3480",
  ],
  gray: [
    "#eff1f5",
    "#e6e9ef",
    "#dce0e8",
    "#ccd0da",
    "#bcc0cc",
    "#9ca0b0",
    "#6c6f85",
    "#5c5f77",
    "#4c4f69",
    "#3e4059",
  ],
  dark: [
    "#cdd6f4",
    "#bac2de",
    "#a6adc8",
    "#7f849c",
    "#45475a",
    "#313244",
    "#26273a",
    "#1e1e2e",
    "#181825",
    "#11111b",
  ],
  primaryShade: { light: 5, dark: 3 },
};

// ── Rosé Pine ── Dawn (light) + Main (dark), muted "natural pine"; pine accent.
export const rosePine: Palette = {
  name: "Rosé Pine",
  white: "#faf4ed",
  black: "#575279",
  accent: [
    "#e7eef1",
    "#c9dbe1",
    "#9fbfc9",
    "#6fa0ae",
    "#478594",
    "#31748f",
    "#2b6b84",
    "#286983",
    "#21596f",
    "#1a4757",
  ],
  gray: [
    "#fffaf3",
    "#f4ede8",
    "#e8e0dc",
    "#dfdad9",
    "#cecacd",
    "#9893a5",
    "#797593",
    "#6e6a86",
    "#575279",
    "#453f5e",
  ],
  dark: [
    "#e0def4",
    "#cecdda",
    "#908caa",
    "#6e6a86",
    "#403d52",
    "#2f2b43",
    "#26233a",
    "#191724",
    "#14121d",
    "#0f0e16",
  ],
  primaryShade: { light: 7, dark: 5 },
};

// ── Gruvbox ── warm retro cream (light) + charcoal (dark), higher contrast.
export const gruvbox: Palette = {
  name: "Gruvbox",
  white: "#fbf1c7",
  black: "#3c3836",
  accent: [
    "#e9f0ef",
    "#caddda",
    "#a4c3bf",
    "#79a49f",
    "#588b86",
    "#458588",
    "#3d7679",
    "#076678",
    "#0a5560",
    "#094650",
  ],
  gray: [
    "#fbf1c7",
    "#f2e5bc",
    "#ebdbb2",
    "#d5c4a1",
    "#bdae93",
    "#a89984",
    "#7c6f64",
    "#665c54",
    "#504945",
    "#3c3836",
  ],
  dark: [
    "#ebdbb2",
    "#d5c4a1",
    "#a89984",
    "#7c6f64",
    "#504945",
    "#3c3836",
    "#333030",
    "#282828",
    "#1d2021",
    "#161616",
  ],
  primaryShade: { light: 7, dark: 3 },
};

export const palettes = { solarized, catppuccin, rosePine, gruvbox };

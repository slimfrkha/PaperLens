/** Local inline-SVG icons — no icon dependency. Stroke-based, inherit currentColor,
 *  aria-hidden (decorative). Sized via the `size` prop (default 18). */
type IconProps = { size?: number | string; className?: string; style?: React.CSSProperties };

function Svg({
  size = 18,
  children,
  ...rest
}: IconProps & { children: React.ReactNode; fill?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{ display: "block", flexShrink: 0 }}
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconSidebar = (p: IconProps) => (
  <Svg {...p}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M9 4v16" />
  </Svg>
);

export const IconPlus = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 5v14M5 12h14" />
  </Svg>
);

export const IconTrash = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13" />
  </Svg>
);

export const IconSearch = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </Svg>
);

export const IconSend = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 12h13M12 5l7 7-7 7" />
  </Svg>
);

export const IconSpark = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 3l1.6 4.6L18 9l-4.4 1.4L12 15l-1.6-4.6L6 9l4.4-1.4z" />
    <path d="M18 14l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z" />
  </Svg>
);

export const IconExternal = (p: IconProps) => (
  <Svg {...p}>
    <path d="M14 5h5v5M19 5l-8 8M12 5H6a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-6" />
  </Svg>
);

export const IconChevron = (p: IconProps) => (
  <Svg {...p}>
    <path d="m9 6 6 6-6 6" />
  </Svg>
);

export const IconRescan = (p: IconProps) => (
  <Svg {...p}>
    <path d="M20 11a8 8 0 1 0-.9 3.7" />
    <path d="M20 5v6h-6" />
  </Svg>
);

export const IconSun = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </Svg>
);

export const IconMoon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
  </Svg>
);

export const IconEdit = (p: IconProps) => (
  <Svg {...p}>
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" />
  </Svg>
);

export const IconCheck = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 13l4 4L19 7" />
  </Svg>
);

export const IconX = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6 6l12 12M18 6l-12 12" />
  </Svg>
);

export const IconBook = (p: IconProps) => (
  <Svg {...p}>
    <path d="M4 5a2 2 0 0 1 2-2h9a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H6a2 2 0 0 0-2 2z" />
    <path d="M4 19a2 2 0 0 1 2-2h10" />
  </Svg>
);

/** `filled`, when given, is a CSS color used as the solid fill (e.g. a Mantine
 *  `--mantine-color-*-filled` token) — pass a fixed, scheme-aware token here rather
 *  than `currentColor`; `currentColor` inherits the ActionIcon "subtle" variant's
 *  *text* shade, which is tuned for a thin stroke, not a large solid fill, and reads
 *  washed-out once filled in dark mode. */
export const IconThumbUp = ({ filled, ...p }: IconProps & { filled?: string }) => (
  <Svg {...p} fill={filled ?? "none"}>
    <path d="M7 10v11" />
    <path d="M7 10l4-7a2 2 0 0 1 2 2v4h5a2 2 0 0 1 2 2.2l-1.2 8A2 2 0 0 1 17 21H7" />
  </Svg>
);

export const IconThumbDown = ({ filled, ...p }: IconProps & { filled?: string }) => (
  <Svg {...p} fill={filled ?? "none"}>
    <path d="M17 14V3" />
    <path d="M17 14l-4 7a2 2 0 0 1-2-2v-4H6a2 2 0 0 1-2-2.2l1.2-8A2 2 0 0 1 7 3h10" />
  </Svg>
);

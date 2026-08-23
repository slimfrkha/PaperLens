import {
  ActionIcon,
  AppShell,
  Group,
  Text,
  Tooltip,
  useComputedColorScheme,
  useMantineColorScheme,
} from "@mantine/core";
import { NavLink, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import PapersPage from "./pages/PapersPage";
import PaperViewer from "./pages/PaperViewer";
import NotesPage from "./pages/NotesPage";
import AdminPage from "./pages/AdminPage";
import { IconBook, IconMoon, IconSun } from "./components/Icons";

/** Light/dark toggle. Initial scheme stays "auto" (follows the OS) until the
 *  user flips it; Mantine then persists the explicit choice in localStorage. */
function ColorSchemeToggle() {
  const { setColorScheme } = useMantineColorScheme();
  const computed = useComputedColorScheme("light", { getInitialValueInEffect: true });
  const dark = computed === "dark";
  return (
    <Tooltip label={dark ? "Light mode" : "Dark mode"}>
      <ActionIcon
        variant="subtle"
        color="gray"
        aria-label="Toggle color scheme"
        onClick={() => setColorScheme(dark ? "light" : "dark")}
      >
        {dark ? <IconSun size={18} /> : <IconMoon size={18} />}
      </ActionIcon>
    </Tooltip>
  );
}

function Nav({ to, label }: { to: string; label: string }) {
  return (
    <NavLink to={to} end={to === "/"} style={{ textDecoration: "none" }}>
      {({ isActive }) => (
        <Text
          span
          fz="sm"
          fw={isActive ? 600 : 450}
          style={{
            color: isActive
              ? "var(--mantine-color-accent-light-color)"
              : "var(--mantine-color-dimmed)",
            paddingBottom: 4,
            borderBottom: `2px solid ${isActive ? "var(--mantine-color-accent-filled)" : "transparent"}`,
            transition: "color 120ms ease, border-color 120ms ease",
          }}
        >
          {label}
        </Text>
      )}
    </NavLink>
  );
}

export default function App() {
  return (
    <AppShell header={{ height: 60 }} padding="lg">
      <AppShell.Header
        withBorder
        style={{
          backdropFilter: "saturate(180%) blur(10px)",
          background: "color-mix(in srgb, var(--mantine-color-body) 82%, transparent)",
        }}
      >
        <Group
          h="100%"
          px="xl"
          justify="space-between"
          style={{ maxWidth: 1180, margin: "0 auto" }}
        >
          <Group gap={9} align="center">
            <IconBook size={20} />
            <Text
              fz="lg"
              fw={500}
              ff="'Newsreader', Georgia, serif"
              style={{ letterSpacing: "-0.01em" }}
            >
              PaperLens
            </Text>
          </Group>
          <Group gap="lg" align="center">
            <Group gap="xl">
              <Nav to="/" label="Chat" />
              <Nav to="/papers" label="Papers" />
              <Nav to="/notes" label="Notes" />
              <Nav to="/admin" label="Admin" />
            </Group>
            <ColorSchemeToggle />
          </Group>
        </Group>
      </AppShell.Header>
      <AppShell.Main>
        <div style={{ maxWidth: 1180, margin: "0 auto" }}>
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/c/:chatId" element={<ChatPage />} />
            <Route path="/papers" element={<PapersPage />} />
            <Route path="/papers/:id" element={<PaperViewer />} />
            <Route path="/notes" element={<NotesPage />} />
            <Route path="/admin" element={<AdminPage />} />
          </Routes>
        </div>
      </AppShell.Main>
    </AppShell>
  );
}

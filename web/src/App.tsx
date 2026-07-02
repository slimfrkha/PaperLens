import { AppShell, Group, Title } from "@mantine/core";
import { NavLink, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import PapersPage from "./pages/PapersPage";
import PaperViewer from "./pages/PaperViewer";
import AdminPage from "./pages/AdminPage";

function Nav({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      style={({ isActive }) => ({
        textDecoration: "none",
        fontWeight: 500,
        color: isActive ? "var(--mantine-color-blue-6)" : "var(--mantine-color-dimmed)",
      })}
    >
      {label}
    </NavLink>
  );
}

export default function App() {
  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Title order={4}>📚 PaperLens</Title>
          <Group gap="lg">
            <Nav to="/" label="Chat" />
            <Nav to="/papers" label="Papers" />
            <Nav to="/admin" label="Admin" />
          </Group>
        </Group>
      </AppShell.Header>
      <AppShell.Main>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/c/:chatId" element={<ChatPage />} />
          <Route path="/papers" element={<PapersPage />} />
          <Route path="/papers/:id" element={<PaperViewer />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </AppShell.Main>
    </AppShell>
  );
}

import { ActionIcon, Button, Group, Paper, ScrollArea, Stack, Text } from "@mantine/core";
import type { ChatSummary } from "../api";

export default function ChatSidebar({
  sessions,
  activeId,
  onNew,
  onSelect,
  onDelete,
}: {
  sessions: ChatSummary[];
  activeId?: string;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <Paper withBorder w={250} p="sm" style={{ flexShrink: 0, alignSelf: "stretch" }}>
      <Button fullWidth mb="sm" variant="light" onClick={onNew}>
        + New chat
      </Button>
      <ScrollArea.Autosize mah="calc(100vh - 160px)">
        <Stack gap={2}>
          {sessions.length === 0 && (
            <Text size="xs" c="dimmed" ta="center" mt="sm">
              No chats yet
            </Text>
          )}
          {sessions.map((s) => (
            <Group
              key={s.id}
              justify="space-between"
              wrap="nowrap"
              gap={4}
              onClick={() => onSelect(s.id)}
              style={{
                padding: "6px 8px",
                borderRadius: 6,
                cursor: "pointer",
                background:
                  s.id === activeId ? "var(--mantine-color-blue-light)" : undefined,
              }}
            >
              <Text size="sm" lineClamp={1} style={{ flex: 1 }}>
                {s.name || "New chat"}
              </Text>
              <ActionIcon
                size="sm"
                variant="subtle"
                color="red"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(s.id);
                }}
              >
                ×
              </ActionIcon>
            </Group>
          ))}
        </Stack>
      </ScrollArea.Autosize>
    </Paper>
  );
}

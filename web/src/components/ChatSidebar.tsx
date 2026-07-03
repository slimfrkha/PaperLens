import { ActionIcon, Box, Button, Group, ScrollArea, Stack, Text } from "@mantine/core";
import { IconPlus, IconTrash } from "./Icons";
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
    <Box
      w={252}
      p="xs"
      style={{
        flexShrink: 0,
        alignSelf: "stretch",
        borderRight: "1px solid var(--pl-border)",
      }}
    >
      <Button
        fullWidth
        mb="sm"
        variant="light"
        leftSection={<IconPlus size={16} />}
        onClick={onNew}
      >
        New chat
      </Button>
      <ScrollArea.Autosize mah="calc(100vh - 160px)" type="hover">
        <Stack gap={2}>
          {sessions.length === 0 && (
            <Text size="xs" c="dimmed" ta="center" mt="md">
              No conversations yet
            </Text>
          )}
          {sessions.map((s) => {
            const active = s.id === activeId;
            return (
              <Group
                key={s.id}
                className="chat-row"
                justify="space-between"
                wrap="nowrap"
                gap={4}
                onClick={() => onSelect(s.id)}
                style={{
                  padding: "7px 9px",
                  borderRadius: 8,
                  cursor: "pointer",
                  background: active ? "var(--mantine-color-accent-light)" : undefined,
                  boxShadow: active
                    ? "inset 2px 0 0 var(--mantine-color-accent-filled)"
                    : undefined,
                }}
              >
                <Text
                  size="sm"
                  lineClamp={1}
                  fw={active ? 500 : 400}
                  style={{
                    flex: 1,
                    color: active ? "var(--mantine-color-accent-light-color)" : undefined,
                  }}
                >
                  {s.name || "New chat"}
                </Text>
                <ActionIcon
                  size="sm"
                  variant="subtle"
                  color="gray"
                  aria-label="Delete chat"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(s.id);
                  }}
                >
                  <IconTrash size={15} />
                </ActionIcon>
              </Group>
            );
          })}
        </Stack>
      </ScrollArea.Autosize>
    </Box>
  );
}

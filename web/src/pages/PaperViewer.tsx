import { useEffect, useRef, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import { Anchor, Badge, Box, Center, Group, Loader, Stack, Title } from "@mantine/core";
import Markdown from "../components/Markdown";
import { IconExternal } from "../components/Icons";
import { getPaper } from "../api";
import { clearHighlight, highlightPassage } from "../highlight";

type PaperData = Awaited<ReturnType<typeof getPaper>>;

export default function PaperViewer() {
  const { id } = useParams();
  const location = useLocation() as { state?: { highlight?: string; section?: string } };
  const [data, setData] = useState<PaperData | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (id) getPaper(id).then(setData);
    return clearHighlight;
  }, [id]);

  useEffect(() => {
    const snippet = location.state?.highlight;
    if (data && snippet && ref.current) {
      const el = ref.current;
      const t = setTimeout(() => highlightPassage(el, snippet), 200);
      return () => clearTimeout(t);
    }
  }, [data, location.state]);

  if (!data)
    return (
      <Center mih="60vh">
        <Loader color="accent" />
      </Center>
    );

  return (
    <Center>
      <Stack gap="lg" w="100%" maw={760} py="md">
        <Stack gap="sm">
          <Title order={1} lh={1.15}>
            {data.title}
          </Title>
          <Group gap={8}>
            {data.arxiv_id && (
              <Anchor
                href={`https://arxiv.org/abs/${data.arxiv_id}`}
                target="_blank"
                rel="noreferrer"
                size="sm"
              >
                <Group gap={4} wrap="nowrap">
                  arXiv:{data.arxiv_id}
                  <IconExternal size={13} />
                </Group>
              </Anchor>
            )}
            {data.tags.map((t) => (
              <Badge key={t} variant="light" color="gray" size="xs" radius="sm" fw={500}>
                {t}
              </Badge>
            ))}
          </Group>
        </Stack>
        <Box ref={ref} className="reading">
          <Markdown>{data.markdown}</Markdown>
        </Box>
      </Stack>
    </Center>
  );
}

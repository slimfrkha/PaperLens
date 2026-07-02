import { useEffect, useRef, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import { Anchor, Badge, Container, Group, Loader, Title } from "@mantine/core";
import Markdown from "../components/Markdown";
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
      <Container>
        <Loader />
      </Container>
    );

  return (
    <Container size="md">
      <Title order={2}>{data.title}</Title>
      <Group gap={6} my="sm">
        {data.arxiv_id && (
          <Anchor href={`https://arxiv.org/abs/${data.arxiv_id}`} target="_blank" size="sm">
            arXiv:{data.arxiv_id}
          </Anchor>
        )}
        {data.tags.map((t) => (
          <Badge key={t} variant="light" size="xs">
            {t}
          </Badge>
        ))}
      </Group>
      <div ref={ref}>
        <Markdown>{data.markdown}</Markdown>
      </div>
    </Container>
  );
}

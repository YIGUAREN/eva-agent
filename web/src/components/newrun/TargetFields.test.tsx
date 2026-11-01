import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TargetFields } from "./TargetFields";

const targets = [
  { name: "demo", kind: "openai_compatible", provider: "demo", model: "demo-echo" },
];

describe("TargetFields", () => {
  it("renders targets, prompt count and errors", () => {
    const html = renderToStaticMarkup(
      <TargetFields
        targets={targets}
        loading={false}
        targetRef="demo"
        prompts={"第一行\n\n第二行"}
        targetRefError="请选择目标"
        promptsError="最多 100 条提示词"
        onTargetChange={() => {}}
        onPromptsChange={() => {}}
      />,
    );

    expect(html).toContain("请选择目标");
    expect(html).toContain("demo · openai_compatible · demo-echo");
    expect(html).toContain("已填写 2 条");
    expect(html).toContain("最多 100 条提示词");
    expect(html).toContain("单条不超过 16384 字节");
  });

  it("renders the loading state", () => {
    const html = renderToStaticMarkup(
      <TargetFields
        targets={[]}
        loading
        targetRef=""
        prompts=""
        onTargetChange={() => {}}
        onPromptsChange={() => {}}
      />,
    );

    expect(html).toContain("加载目标中…");
  });

  it("renders the empty state instead of the loading hint", () => {
    const html = renderToStaticMarkup(
      <TargetFields
        targets={[]}
        loading={false}
        targetRef=""
        prompts=""
        onTargetChange={() => {}}
        onPromptsChange={() => {}}
      />,
    );

    expect(html).toContain("暂无可用目标，请检查 config/targets.yaml");
    expect(html).not.toContain("加载目标中…");
  });
});

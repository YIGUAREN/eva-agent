import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { PasteFields } from "./PasteFields";

describe("PasteFields", () => {
  it("renders the three fields, their values and the error", () => {
    const html = renderToStaticMarkup(
      <PasteFields
        value={{ query: "问题", response: "回答", context: "" }}
        queryError="请填写问题 / 提示词"
        onChange={() => {}}
      />,
    );

    expect(html).toContain("问题 / 提示词");
    expect(html).toContain("待评估回答");
    expect(html).toContain("背景（可选）");
    expect(html).toContain(">问题</textarea>");
    expect(html).toContain(">回答</textarea>");
    expect(html).toContain("请填写问题 / 提示词");
  });
});

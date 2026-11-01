import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { DatasetFields } from "./DatasetFields";

describe("DatasetFields", () => {
  it("renders the path, formats, column map and errors", () => {
    const html = renderToStaticMarkup(
      <DatasetFields
        value={{ path: "data/qa.csv", format: "jsonl", columnMap: "query=q" }}
        pathError="请填写数据集文件路径"
        columnMapError="列映射格式应为 query=q,response=a"
        onChange={() => {}}
      />,
    );

    expect(html).toContain("数据集文件路径");
    expect(html).toContain('value="data/qa.csv"');
    expect(html).toContain("CSV");
    expect(html).toContain("JSONL");
    expect(html).toContain("列映射（可选）");
    expect(html).toContain('value="query=q"');
    expect(html).toContain("请填写数据集文件路径");
    expect(html).toContain("列映射格式应为 query=q,response=a");
  });
});

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { UploadField } from "./UploadField";

describe("UploadField", () => {
  it("renders the accept filter, selected file and error", () => {
    const file = { name: "evidence.zip", size: 2048 } as unknown as File;
    const html = renderToStaticMarkup(
      <UploadField file={file} error="请选择上传包" onChange={() => {}} />,
    );

    expect(html).toContain('accept=".zip,.tar,.tar.gz,.tgz"');
    expect(html).toContain("已选择：evidence.zip（2.0 KB）");
    expect(html).toContain("请选择上传包");
    expect(html).toContain("sandbox-runner");
  });

  it("renders no selection summary without a file", () => {
    const html = renderToStaticMarkup(<UploadField file={null} onChange={() => {}} />);

    expect(html).not.toContain("已选择");
  });
});

import { describe, expect, it } from "vitest";
import { bundleForm } from "./MarketDatasets";
describe("offline research bundle upload", () => {
  it("uses only bounded named multipart fields and preserves missing files for incomplete preview", () => {
    const metadata = new File(["{}"], "metadata.json", {
      type: "application/json",
    });
    const form = bundleForm({
      metadata,
      arbitrary: new File(["secret"], "other.csv"),
    });
    expect([...form.keys()]).toEqual(["metadata"]);
    expect((form.get("metadata") as File).name).toBe("metadata.json");
    expect(form.has("actions")).toBe(false);
  });
});

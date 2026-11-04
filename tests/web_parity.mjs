// Drives handformer/viz/static/local-api.js under Node for tests/test_web.py.
//   node --experimental-detect-module tests/web_parity.mjs MODEL_DIR REQUESTS.json OUT.json
// REQUESTS: [[path, body], ...]; "$run" inside a body is replaced by the id of
// the most recent /api/run. Typed arrays are written as plain lists.
import { readFileSync, writeFileSync } from "node:fs";
import { LocalApi } from "../handformer/viz/static/local-api.js";

const [dir, reqFile, outFile] = process.argv.slice(2);
const api = new LocalApi(JSON.parse(readFileSync(`${dir}/model.json`, "utf8")),
                         JSON.parse(readFileSync(`${dir}/info.json`, "utf8")));
let runId = null;
const out = [];
for (const [path, body] of JSON.parse(readFileSync(reqFile, "utf8"))) {
  const b = body && Object.fromEntries(Object.entries(body).map(([k, v]) => [k, v === "$run" ? runId : v]));
  try {
    const r = await api.call(path, b);
    if (path === "/api/run") runId = r.run_id;
    out.push(r && r.data ? { meta: r.meta, data: Array.from(r.data) } : r);
  } catch (e) {
    out.push({ error: e.message });
  }
}
writeFileSync(outFile, JSON.stringify(out));
console.info(`[web_parity] answered ${out.length} requests`);

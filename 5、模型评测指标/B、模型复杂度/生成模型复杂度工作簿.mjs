import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const packageBase = process.env.CODEX_SPREADSHEET_DEPS || "C:/Users/sfr08/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/package.json";
const packageRequire = createRequire(packageBase);
const { SpreadsheetFile, Workbook } = packageRequire("@oai/artifact-tool");

const dir = path.dirname(fileURLToPath(import.meta.url));
const report = JSON.parse(await fs.readFile(path.join(dir, "模型复杂度完整结果.json"), "utf8"));
const outDir = path.join(dir, "Charts and pictures");
const output = path.join(outDir, "H2_vs_H2-plus_模型复杂度对比.xlsx");
const workbook = Workbook.create();
const summary = workbook.worksheets.add("总览");
const detail = workbook.worksheets.add("逐序列");
summary.showGridLines = false;
detail.showGridLines = false;
summary.tabColor = "#165D8D";
detail.tabColor = "#30A1A8";

const navy = "#17324F";
const teal = "#087E8B";
const pale = "#EAF5F6";
const soft = "#F2F6FA";
const ink = "#183149";
const font = "Microsoft YaHei";

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: navy,
    font: { name: font, color: "#FFFFFF", bold: true, size: 10 },
    rowHeight: 30,
    verticalAlignment: "center",
  };
}
function model(role, name) {
  return report.results[role].models[name];
}

summary.getRange("A1:G1").format = {
  fill: navy,
  font: { name: font, color: "#FFFFFF", bold: true, size: 17 },
  rowHeight: 44,
};
summary.getRange("A1").values = [["H2 / H2-plus  算法复杂度正式比较"]];
summary.getRange("A2").values = [["冻结 exact INT8；Validation / Test 各 24 条 × 8 s。复杂度较低不直接代表检测效果更好，需结合 A 的准确度指标。"]];
summary.getRange("A2:G2").format = { fill: soft, font: { name: font, color: ink, size: 10 }, rowHeight: 30 };
summary.getRange("A4:G4").values = [["主表编号及指标", "单位", "Validation · H2", "Validation · H2-plus", "Test · H2", "Test · H2-plus", "口径 / 注意事项"]];
styleHeader(summary, "A4:G4");

const metrics = [
  ["A01  参数量", "个", "total_weight_parameters", "含零权重；共享 Proposal 只计一次"],
  ["A02  静态张量载荷", "byte", "total_tensor_payload_bytes", "INT8 权重 + FP32 逐输出通道 scale；不含模型文件包装"],
  ["A03  持久神经元状态", "byte", "persistent_neuron_state_bytes", "两模型均为 ANN；不是 SNN 神经元状态"],
  ["A03  核心张量预算上限", "byte", "max_core_tensor_bytes_observed", "逐序列核心张量体积之最大值；非同时存活峰值"],
  ["A03  软件追踪堆峰值", "byte", "work_memory_peak_bytes", "只回放本划分事件最多序列；不含原始输入/已加载权重；非 RSS/SRAM"],
  ["A04  稠密 MAC · 总量", "MAC", "total_dense_macs", "Proposal 全窗 + 实际候选事件的 Refiner"],
  ["A04  稠密 MAC · 输入每秒", "MAC/s", "dense_macs_per_second_input", "总量 / 192 秒；不是实测处理速度"],
  ["A05  有效 MAC · 总量", "MAC", "total_effective_macs", "非零输入激活 × 非零权重连接；仅潜在稀疏执行量"],
  ["A05  有效 MAC · 输入每秒", "MAC/s", "effective_macs_per_second_input", "总量 / 192 秒；当前软件仍执行稠密乘法"],
  ["A05  实际 SNN AC", "AC", "effective_acs", "当前 ANN 不存在脉冲累加"],
  ["A06  SNN 神经元更新", "次", "snn_neuron_updates", "当前 ANN 不适用，记 0 次"],
  ["A06  隐藏层量化变换", "元素", "activation_transform_elements", "整数路径逐层变换；不与 MAC 直接相加"],
  ["A06  Proposal sigmoid", "元素", "proposal_sigmoid_elements", "Proposal 输出元素数"],
  ["A06  Refiner softmax", "候选事件", "refiner_softmax_items", "与候选事件数相同"],
  ["A06  历史特征值", "元素", "history_feature_values", "只在 H2-plus 计算；不等于神经元更新"],
];
const roles = [["Validation", "H2"], ["Validation", "H2-plus"], ["Test", "H2"], ["Test", "H2-plus"]];
const metricRows = metrics.map(([label, unit, key, note]) => [
  label, unit, ...roles.map(([role, name]) => model(role, name)[key]), note,
]);
summary.getRangeByIndexes(4, 0, metricRows.length, 7).values = metricRows;
summary.getRange(`A5:G${4 + metricRows.length}`).format.font = { name: font, size: 10, color: ink };
summary.getRange(`C5:F${4 + metricRows.length}`).format.numberFormat = "#,##0";
for (const row of [5, 6, 8, 10, 12, 15, 16]) {
  summary.getRange(`A${row}:G${row}`).format.fill = pale;
}
summary.getRange("A21").values = [["解读：H2-plus 以 +256 个参数换取 A 中更高的目标检测表现；复杂度指标应与 A 并列，不单独判定优劣。"]];
summary.getRange("A21:G21").format = { fill: soft, font: { name: font, size: 10, color: teal, bold: true }, rowHeight: 30 };
summary.getRange("A23").values = [["源结果：模型复杂度完整结果.json；权重 SHA-256 和数据索引由生成模型复杂度评测.py 核验。"]];
summary.getRange("A24").values = [["软件追踪峰值仅针对各划分事件最多的一条序列；其它全部指标覆盖各划分 24 条序列。"]];
summary.getRange("A25").values = [["有效 MAC 是理论稀疏活动量，不等于 CPU/GPU 实际指令数、芯片功耗或上板延迟。"]];
summary.getRange("A23:G25").format.font = { name: font, size: 9, color: "#576B7C" };
summary.getRange("A:A").format.columnWidth = 30;
summary.getRange("B:B").format.columnWidth = 16;
summary.getRange("C:F").format.columnWidth = 23;
summary.getRange("G:G").format.columnWidth = 76;
summary.freezePanes.freezeRows(4);

detail.getRange("A1:P1").format = { fill: navy, font: { name: font, color: "#FFFFFF", bold: true, size: 15 }, rowHeight: 42 };
detail.getRange("A1").values = [["逐序列 · 算法复杂度溯源"]];
detail.getRange("A2").values = [["每条冻结 8 s 序列；逐序列事件数和候选数已与 A 的 exact INT8 per_sequence.csv 对齐核验。"]];
detail.getRange("A2:P2").format = { fill: soft, font: { name: font, color: ink, size: 10 }, rowHeight: 29 };
const headers = ["划分", "模型", "序列", "事件数", "候选数", "Proposal 调用", "Proposal 稠密 MAC", "Refiner 稠密 MAC", "合计稠密 MAC", "Proposal 有效 MAC", "Refiner 有效 MAC", "合计有效 MAC", "量化变换元素", "sigmoid 元素", "历史特征值", "核心张量预算 byte"];
detail.getRange("A4:P4").values = [headers];
styleHeader(detail, "A4:P4");
const rows = [];
for (const role of ["Validation", "Test"]) {
  for (const name of ["H2", "H2-plus"]) {
    for (const r of report.results[role].per_sequence[name]) {
      rows.push([role, name, r.sequence, r.events, r.candidates, r.proposal_calls,
        r.proposal_dense_macs, r.refiner_dense_macs, r.proposal_dense_macs + r.refiner_dense_macs,
        r.proposal_effective_macs, r.refiner_effective_macs,
        r.proposal_effective_macs + r.refiner_effective_macs, r.activation_transform_elements,
        r.proposal_sigmoid_elements, r.history_feature_values, r.core_tensor_bytes]);
    }
  }
}
detail.getRangeByIndexes(4, 0, rows.length, headers.length).values = rows;
detail.getRangeByIndexes(4, 0, rows.length, headers.length).format.font = { name: font, size: 10, color: ink };
detail.getRange(`D5:P${rows.length + 4}`).format.numberFormat = "#,##0";
detail.getRange("A:A").format.columnWidth = 17;
detail.getRange("B:B").format.columnWidth = 14;
detail.getRange("C:C").format.columnWidth = 19;
detail.getRange("D:F").format.columnWidth = 17;
detail.getRange("G:P").format.columnWidth = 24;
detail.freezePanes.freezeRows(4);

workbook.recalculate();
const audit = await workbook.inspect({ kind: "region", sheetId: "总览", range: "A4:F10", maxChars: 1800, tableMaxRows: 7, tableMaxCols: 6 });
console.log(audit.ndjson);
await fs.mkdir(outDir, { recursive: true });
const blob = await SpreadsheetFile.exportXlsx(workbook);
await blob.save(output);
console.log(output);

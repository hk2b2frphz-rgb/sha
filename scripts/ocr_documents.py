#!/usr/bin/env python3
"""
文書 (PDF / PPTX / DOCX / 画像) を VLM (Qwen3-VL) で OCR してテキスト化する。

スキャン PDF やパワポの図表内テキストも読み取れる。
出力は 1 文書 = 1 テキストファイル。extract_terms.py にそのまま渡せる。

処理:
  1. PPTX / DOCX は LibreOffice (soffice) で PDF に変換
  2. PDF は pypdfium2 でページごとに画像化
  3. 各ページ画像を Qwen3-VL に渡して文章を書き起こす

使い方 (A100 など GPU マシンで):
  uv run --project gemma_runtime python scripts/ocr_documents.py \
      --docs docs/ --out-dir out/text

要件: PPTX/DOCX を扱う場合は LibreOffice が必要 (sudo apt install libreoffice)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ocr_documents")

OCR_PROMPT = (
    "この画像に含まれる文章をすべて書き起こしてください。"
    "図表の中のテキストも含めてください。"
    "書き起こしのみを出力し、説明やコメントは不要です。"
    "文章が含まれていない場合は「(テキストなし)」とだけ出力してください。"
)

DOC_SUFFIXES = {".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文書を VLM で OCR してテキスト化する")
    parser.add_argument(
        "--docs", type=Path, nargs="+", required=True,
        help="文書ファイルまたはディレクトリ (pdf/pptx/docx/png/jpg, 複数指定可)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="テキスト出力先")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="auto")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="PDF レンダリング倍率 (2.0 ≒ 144dpi)。小さい文字が潰れるなら上げる")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 入力収集と PDF 化
# ---------------------------------------------------------------------------

def collect_docs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for p in inputs:
        if p.is_dir():
            paths.extend(
                f for f in sorted(p.rglob("*")) if f.suffix.lower() in DOC_SUFFIXES
            )
        elif p.suffix.lower() in DOC_SUFFIXES:
            paths.append(p)
        else:
            logger.warning("非対応形式のためスキップ: %s", p)
    if not paths:
        raise SystemExit(f"対象文書が見つかりません (対応形式: {', '.join(sorted(DOC_SUFFIXES))})")
    return paths


def office_to_pdf(path: Path, work_dir: Path) -> Path:
    """PPTX/DOCX を LibreOffice で PDF に変換する。"""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        raise SystemExit(
            f"{path.name}: PPTX/DOCX の変換には LibreOffice が必要です。\n"
            "  sudo apt install libreoffice  (または該当パッケージ) を実行してください。"
        )
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(work_dir), str(path)],
        check=True, capture_output=True, timeout=300,
    )
    pdf = work_dir / (path.stem + ".pdf")
    if not pdf.exists():
        raise RuntimeError(f"{path.name}: PDF 変換に失敗しました")
    return pdf


def pdf_to_images(path: Path, scale: float) -> list[Any]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    images = []
    for page in pdf:
        images.append(page.render(scale=scale).to_pil())
    return images


# ---------------------------------------------------------------------------
# VLM OCR
# ---------------------------------------------------------------------------

class VlmOcr:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype: Any = "auto"
        if self.args.dtype != "auto":
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[self.args.dtype]

        logger.info("VLM をロード中: %s", self.args.model)
        start = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(self.args.model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.args.model, dtype=dtype, device_map=self.args.device_map
        )
        logger.info("ロード完了 (%.1f 秒)", time.monotonic() - start)

    def ocr_image(self, image: Any) -> str:
        self.load()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        import torch
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False
            )
        text = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        if text == "(テキストなし)":
            return ""
        return text


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    docs = collect_docs(args.docs)
    logger.info("%d 個の文書を処理します", len(docs))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ocr = VlmOcr(args)
    failed: list[str] = []
    start = time.monotonic()

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for di, doc in enumerate(docs, 1):
            logger.info("--- (%d/%d) %s ---", di, len(docs), doc.name)
            try:
                suffix = doc.suffix.lower()
                if suffix in {".png", ".jpg", ".jpeg"}:
                    from PIL import Image
                    images = [Image.open(doc)]
                else:
                    pdf_path = doc if suffix == ".pdf" else office_to_pdf(doc, work_dir)
                    images = pdf_to_images(pdf_path, args.scale)
            except Exception as exc:
                logger.error("%s: 読み込み失敗 (%s)", doc.name, exc)
                failed.append(doc.name)
                continue

            out_path = args.out_dir / (doc.stem + ".txt")
            pages: list[str] = []
            with out_path.open("w", encoding="utf-8") as fh:
                for pi, image in enumerate(images, 1):
                    text = ocr.ocr_image(image)
                    pages.append(text)
                    fh.write(text + "\n\n")
                    fh.flush()  # 中断してもページ単位で残る
                    logger.info(
                        "  ページ %d/%d: %d 文字 | %s",
                        pi, len(images), len(text),
                        (text[:40].replace("\n", " ") + "...") if text else "(テキストなし)",
                    )
            total_chars = sum(len(p) for p in pages)
            elapsed = time.monotonic() - start
            eta = elapsed / di * (len(docs) - di)
            logger.info(
                "(%d/%d) %s -> %s (%d 文字) | 経過 %.0f 秒 / 残り目安 %.0f 秒",
                di, len(docs), doc.name, out_path.name, total_chars, elapsed, eta,
            )
            if total_chars == 0:
                logger.warning("%s: テキストが抽出できませんでした", doc.name)

    logger.info("完了: %s にテキストを保存", args.out_dir)
    if failed:
        logger.error("失敗した文書: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()

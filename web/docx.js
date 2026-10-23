"use strict";

// Minimal .docx text extraction in the browser: read the zip's central directory, inflate
// word/document.xml with DecompressionStream, then collect paragraph text. No libraries.

async function readDocx(file) {
  const buf = await file.arrayBuffer();
  const view = new DataView(buf);

  // End-of-central-directory record: signature 0x06054b50 within the last 64 KB.
  let eocd = -1;
  for (let i = buf.byteLength - 22; i >= Math.max(0, buf.byteLength - 65557); i--) {
    if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("This file is not a valid .docx document.");

  const entries = view.getUint16(eocd + 10, true);
  let ptr = view.getUint32(eocd + 16, true);
  const decoder = new TextDecoder();
  for (let n = 0; n < entries; n++) {
    if (view.getUint32(ptr, true) !== 0x02014b50) break;
    const method = view.getUint16(ptr + 10, true);
    const compSize = view.getUint32(ptr + 20, true);
    const nameLen = view.getUint16(ptr + 28, true);
    const extraLen = view.getUint16(ptr + 30, true);
    const commentLen = view.getUint16(ptr + 32, true);
    const localOffset = view.getUint32(ptr + 42, true);
    const name = decoder.decode(new Uint8Array(buf, ptr + 46, nameLen));
    if (name === "word/document.xml") {
      const dataStart = localOffset + 30 + view.getUint16(localOffset + 26, true) + view.getUint16(localOffset + 28, true);
      const data = new Uint8Array(buf, dataStart, compSize);
      const xml = method === 0
        ? decoder.decode(data)
        : await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream("deflate-raw"))).text();
      return docxXmlToText(xml);
    }
    ptr += 46 + nameLen + extraLen + commentLen;
  }
  throw new Error("No document text found in this .docx file.");
}

function docxXmlToText(xml) {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  const W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
  const paragraphs = [];
  for (const p of doc.getElementsByTagNameNS(W, "p")) {
    let text = "";
    for (const node of p.getElementsByTagNameNS(W, "*")) {
      if (node.localName === "t") text += node.textContent;
      else if (node.localName === "tab") text += "\t";
      else if (node.localName === "br" || node.localName === "cr") text += "\n";
    }
    if (text.trim()) paragraphs.push(text.trim());
  }
  return paragraphs.join("\n\n");
}

async function readTextFile(file) {
  if (/\.docx$/i.test(file.name)) return readDocx(file);
  if (/\.(txt|md)$/i.test(file.name) || file.type.startsWith("text/")) return file.text();
  throw new Error("Upload a .docx or .txt file. For PDFs, copy the text and paste it in.");
}

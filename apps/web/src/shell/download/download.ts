/** Hand the browser a text file to save. Split from the pure builders so tests never touch Blob URLs. */
export function saveTextFile(name: string, mime: string, text: string, doc: Document = document): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  try {
    const anchor = doc.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.rel = "noopener";
    doc.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } finally {
    // The click has read the URL synchronously; revoke on the next tick to be safe in every browser.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

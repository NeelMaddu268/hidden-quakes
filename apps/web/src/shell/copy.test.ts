/**
 * DEMO-01 acceptance: "a grep finds no numeric literals in UI copy". Every JSX text node, every
 * string literal rendered as a JSX child, and every human-readable string attribute in the shell
 * and the page must be digit-free; numbers reach the screen only through expressions fed by
 * provider data. The scan uses TypeScript's parser, so code, comments and `{...}` expressions are
 * told apart exactly.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { describe, expect, it } from "vitest";

// Resolved with `path`, not `new URL(template, import.meta.url)`: Vite rewrites the latter as an
// asset lookup and hands back `undefined` for a directory.
const HERE = dirname(fileURLToPath(import.meta.url));
const ROOTS = ["shell", "app"].map((dir) => resolve(HERE, "..", dir));

const READABLE_ATTRIBUTES = new Set(["aria-label", "title", "alt", "placeholder", "aria-description"]);

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...walk(path));
    else if (name.endsWith(".tsx") && !name.endsWith(".test.tsx")) out.push(path);
  }
  return out;
}

function normalize(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** Literal text of string and template literals inside an expression (template holes excluded). */
function literalText(node: ts.Node, out: string[]): void {
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
    out.push(node.text);
  } else if (ts.isTemplateExpression(node)) {
    out.push(node.head.text);
    for (const span of node.templateSpans) out.push(span.literal.text);
  }
  ts.forEachChild(node, (child) => literalText(child, out));
}

/** Everything a viewer can read in a TSX source: JSX text, literal JSX children, readable attributes. */
export function readableCopy(source: string, fileName = "copy.tsx"): string[] {
  const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const copy: string[] = [];
  const visit = (node: ts.Node) => {
    if (ts.isJsxText(node)) {
      copy.push(node.text);
    } else if (ts.isJsxExpression(node) && node.expression && !ts.isJsxAttribute(node.parent)) {
      literalText(node.expression, copy);
    } else if (ts.isJsxAttribute(node) && READABLE_ATTRIBUTES.has(node.name.getText(file))) {
      if (node.initializer && ts.isStringLiteral(node.initializer)) copy.push(node.initializer.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return copy.map(normalize).filter(Boolean);
}

describe("UI copy carries no numbers", () => {
  const files = ROOTS.flatMap(walk);

  it("scans the shell and the page", () => {
    expect(files.some((f) => f.endsWith("Shell.tsx"))).toBe(true);
    expect(files.some((f) => f.endsWith("page.tsx"))).toBe(true);
    // The scan actually sees copy: the title is a JSX text node in Shell.tsx.
    const shell = files.find((f) => f.endsWith("Shell.tsx"))!;
    expect(readableCopy(readFileSync(shell, "utf8"))).toContain("Hidden Quakes");
  });

  it("finds no digit in any JSX text node, literal child or readable attribute", () => {
    const offenders: string[] = [];
    for (const file of files) {
      for (const text of readableCopy(readFileSync(file, "utf8"), file)) {
        if (/\d/.test(text)) offenders.push(`${file}: ${JSON.stringify(text)}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it("would catch a literal count in text, in an expression, in a template and in an attribute", () => {
    const text = `export function A() { return <p>{label} 43 events</p>; }`;
    expect(readableCopy(text)).toEqual(["43 events"]);
    const expression = `export function B() { return <p>{ok && "43 events"}</p>; }`;
    expect(readableCopy(expression)).toEqual(["43 events"]);
    const template = "export function C({ n }: { n: number }) { return <p>{`${n} of 43`}</p>; }";
    expect(readableCopy(template)).toEqual(["of 43"]);
    const attribute = `export function D() { return <button aria-label="43 events" />; }`;
    expect(readableCopy(attribute)).toEqual(["43 events"]);
    const fine = "export function E({ n }: { n: number }) { return <p data-testid=\"c-1\">{`${n} events`}</p>; }";
    expect(readableCopy(fine)).toEqual(["events"]);
  });
});

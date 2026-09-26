import type { CSSProperties } from "react";
import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { cssVariables } from "@hq/visualization";
import "./globals.css";

// H3's design tokens (`packages/visualization/tokens.ts`) read exactly these two CSS variables.
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Hidden Quakes",
  description:
    "A public-data-only view of candidate microseismic events beneath Utah's geothermal frontier.",
};

// Every token as a `--hq-*` custom property on <html>, so the shell, the scene and the drawer share
// one palette through CSS. Custom properties are not in React's CSSProperties type, hence the cast.
const tokenStyle = cssVariables() as CSSProperties;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`} style={tokenStyle}>
      <body>{children}</body>
    </html>
  );
}

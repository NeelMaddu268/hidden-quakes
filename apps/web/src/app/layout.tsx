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

const DESCRIPTION =
  "A public-data-only view of candidate microseismic events beneath Utah's geothermal frontier, rebuilt from public seismometers and graded against the public regional catalog.";

// DEMO-04 item 3: link previews. `og.png` is a static frame of the strict view from the run of
// record (apps/web/public/og.png); metadataBase makes its URL absolute for crawlers.
const PREVIEW = { url: "/og.png", width: 1200, height: 630, alt: "Hidden Quakes: strict candidate events underground" };

export const metadata: Metadata = {
  metadataBase: new URL("https://hidden-quakes.vercel.app"),
  title: "Hidden Quakes",
  description: DESCRIPTION,
  openGraph: {
    type: "website",
    url: "/",
    siteName: "Hidden Quakes",
    title: "Hidden Quakes: what the public can't see beneath Utah's geothermal frontier",
    description: DESCRIPTION,
    images: [PREVIEW],
  },
  twitter: {
    card: "summary_large_image",
    title: "Hidden Quakes: what the public can't see beneath Utah's geothermal frontier",
    description: DESCRIPTION,
    images: [PREVIEW.url],
  },
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

import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "CLINICAL TRIAGE | MONITORING STATION V2.0",
  description: "Autonomous Emergency Department Triage Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${jetbrainsMono.variable} font-mono bg-background text-terminal-green scanline min-h-screen overflow-hidden`}>
        {children}
      </body>
    </html>
  );
}

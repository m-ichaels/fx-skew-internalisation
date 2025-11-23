#!/usr/bin/env python3
"""report.pdf from results/summary.md and results/figures/*.png (fpdf2).   python scripts/report.py"""
import os
import re
import sys

from fpdf import FPDF

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "report.pdf"

INTRO = """Question. How should a dealer skew and hedge when some clients are informed, some read the dealer's skew, and hedging on the interbank venue has impact?

Method. A C++ event-driven dealer simulator: reference price from EURUSD ticks (histdata.com, April 2025), client requests as marked point processes on a tiered pricing ladder with logistic spread response, tier-specific post-trade drift (adverse selection), skew readers, and hedging with temporary and permanent impact. Quoters: Avellaneda-Stoikov (naive), the GLFT 2013 closed form (sanity), a quadratic-Hamiltonian closed form with adverse selection, and the Barzykin-Bergault-Gueant 2023 dealer solved numerically (ergodic HJB on an inventory grid) with the 2025 adverse-selection / price-reading extension. On top: an online Bayesian mark-out classifier routing toxic trades to the hedge (Cartea & Sanchez-Betancourt 2025), anticipatory hedging of autocorrelated flow (Nutz-Webster-Zhao 2025) and last look as a control. P&L is decomposed into spread capture, inventory, adverse selection, hedging execution and hedging impact, exactly.

Caveats. Real single-dealer client flow does not exist publicly: the client layer is a calibrated simulation (BIS 2025 turnover shares, Butz-Oomen internalisation horizons, Oomen / LMAX mark-out shapes) and the contribution is reproducing the 2021-25 closed forms and testing the policies on the risk-return frontier."""

FIGS = [("quotes.png", "Ladders over inventory: offsets shown to the informed tier and to readers, and the hedging band."),
        ("skew_by_tier.png", "Skew by tier without and with price reading: readers are shown a flat ladder."),
        ("frontier.png", "Risk-return frontier over risk aversion, with 95% bootstrap bands over days x seeds."),
        ("decomposition.png", "P&L decomposition by quoter (sums by construction)."),
        ("signatures.png", "Price signatures by tier, internalisation holding times and classifier scores."),
        ("regime_sensitivity.png", "Client-flow regime change, latency sensitivity and toxicity-calibration sensitivity."),
        ("lastlook.png", "Last look as a control: P&L and rejection rate by hold time and tolerance.")]


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9); self.set_text_color(120); self.cell(0, 6, "fx-skew-internalisation - FX dealer skew and internalisation model", align="R"); self.ln(8); self.set_text_color(0)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "", 8); self.set_text_color(120); self.cell(0, 6, f"{self.page_no()}", align="C")


def clean(s):
    return (s.replace("–", "-").replace("—", "-").replace("−", "-").replace("σ", "sigma").replace("β", "beta").replace("η", "eta").replace("Δ", "d").replace("×", "x").replace("≥", ">=").replace("…", "...")
             .replace("²", "^2").replace("±", "+/-").replace("**", "").replace("`", "").replace("√", "sqrt ").replace("γ", "gamma").replace("ψ", "psi").replace("μ", "mu").replace("ρ", "rho").replace("λ", "lambda").replace("→", "->").replace("≈", "~"))


def md_table(pdf, rows):
    cols = [c.strip() for c in rows[0].strip("|").split("|")]
    data = [[clean(c.strip()) for c in r.strip("|").split("|")] for r in rows[2:]]
    pdf.set_font("Helvetica", "", 6.5)
    n = len(cols); w = (pdf.w - 20) / n
    widths = [w] * n
    pdf.set_font("Helvetica", "B", 6.5)
    for c, wd in zip(cols, widths): pdf.cell(wd, 5, clean(c)[:40], border=1)
    pdf.ln(5); pdf.set_font("Helvetica", "", 6.5)
    for r in data:
        if pdf.get_y() > pdf.h - 20: pdf.add_page()
        for c, wd in zip(r, widths): pdf.cell(wd, 4.5, c[:40], border=1)
        pdf.ln(4.5)
    pdf.ln(2)


def main():
    pdf = PDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, "FX Dealer Market Making: Skew, Internalisation, Adverse Selection and Price Reading", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for para in INTRO.split("\n\n"):
        pdf.multi_cell(0, 4.5, clean(para)); pdf.ln(2)
    for fn, cap in FIGS:
        p = os.path.join(R, "figures", fn)
        if not os.path.exists(p): continue
        if pdf.get_y() > pdf.h - 90: pdf.add_page()
        pdf.image(p, w=pdf.w - 20); pdf.set_font("Helvetica", "I", 8); pdf.multi_cell(0, 4, clean(cap)); pdf.ln(3); pdf.set_font("Helvetica", "", 9)
    # summary tables
    sm = os.path.join(R, "summary.md")
    if os.path.exists(sm):
        pdf.add_page()
        lines = open(sm, encoding="utf-8").read().splitlines()
        i = 0
        while i < len(lines):
            l = lines[i]
            if l.startswith("## "):
                pdf.set_font("Helvetica", "B", 11); pdf.ln(2); pdf.cell(0, 7, clean(l[3:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("### "):
                pdf.set_font("Helvetica", "B", 9); pdf.cell(0, 6, clean(l[4:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("|"):
                j = i
                while j < len(lines) and lines[j].startswith("|"): j += 1
                if j - i >= 2: md_table(pdf, lines[i:j])
                i = j
            elif l.startswith("```"):
                j = i + 1
                while j < len(lines) and not lines[j].startswith("```"): j += 1
                pdf.set_font("Courier", "", 7)
                for t in lines[i + 1:j]: pdf.set_x(pdf.l_margin); pdf.multi_cell(0, 3.5, clean(t))
                pdf.set_font("Helvetica", "", 9); i = j + 1
            elif l.strip():
                pdf.set_x(pdf.l_margin); pdf.multi_cell(0, 4.5, clean(l.strip())); i += 1
            else:
                i += 1
    pdf.output(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

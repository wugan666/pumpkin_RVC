"""Look and feel of the WebUI (Gradio 3.14 has no theme API, so plain CSS)."""

import html

APP_CSS = """
:root, :host, .gradio-container {
    --rvc-accent: #4f46e5;
    --rvc-accent-strong: #4338ca;
    --rvc-accent-soft: #eef2ff;
    --rvc-border: #e2e8f0;
    --rvc-surface: #ffffff;
    --rvc-page: #f5f7fb;
    --rvc-text: #0f172a;
    --rvc-muted: #64748b;
}
.dark, .dark .gradio-container, .gradio-container.dark {
    --rvc-accent: #818cf8;
    --rvc-accent-strong: #6366f1;
    --rvc-accent-soft: #1e1b4b;
    --rvc-border: #334155;
    --rvc-surface: #111827;
    --rvc-page: #0b1120;
    --rvc-text: #e2e8f0;
    --rvc-muted: #94a3b8;
}
body, gradio-app {
    background: var(--rvc-page) !important;
}
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 12px 20px 40px !important;
    background: transparent !important;
    font-family: "Inter", "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei",
        "Noto Sans CJK SC", system-ui, sans-serif !important;
    color: var(--rvc-text);
}

/* ---- header ---- */
.rvc-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    padding: 18px 22px;
    border-radius: 14px;
    color: #fff;
    background: linear-gradient(120deg, #4338ca 0%, #6366f1 55%, #0ea5e9 100%);
    box-shadow: 0 8px 24px rgba(67, 56, 202, .22);
}
.rvc-header h1 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: .2px;
    color: #fff;
}
.rvc-header p {
    margin: 4px 0 0;
    font-size: 13px;
    opacity: .9;
    color: #fff;
}
.rvc-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.rvc-chip {
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 12px;
    background: rgba(255, 255, 255, .18);
    border: 1px solid rgba(255, 255, 255, .28);
    white-space: nowrap;
}
.rvc-license {
    margin: 8px 2px 0;
    font-size: 12px;
    line-height: 1.5;
    color: var(--rvc-muted);
}

/* ---- tabs ---- */
/* Gradio 3.14 renders the tab bar as div.flex.border-b-2 > button; the
   selected tab is the one without .border-transparent. */
div.flex.border-b-2.flex-wrap {
    border-bottom: 1px solid var(--rvc-border) !important;
    gap: 2px;
}
div.flex.border-b-2.flex-wrap > button {
    margin-bottom: -1px !important;
    padding: 9px 16px !important;
    font-size: 14px !important;
    font-weight: 500;
    color: var(--rvc-muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 8px 8px 0 0 !important;
    background: transparent !important;
    transition: color .15s ease, background .15s ease;
}
div.flex.border-b-2.flex-wrap > button:hover {
    color: var(--rvc-accent) !important;
    background: var(--rvc-accent-soft) !important;
}
div.flex.border-b-2.flex-wrap > button:not(.border-transparent) {
    color: var(--rvc-accent) !important;
    font-weight: 600;
    border-bottom: 2px solid var(--rvc-accent) !important;
    background: transparent !important;
}
.tabitem {
    border: none !important;
    padding: 16px 0 0 !important;
    background: transparent !important;
}

/* ---- surfaces ---- */
.gr-box, .gr-panel, .gr-form {
    border-color: var(--rvc-border) !important;
    border-radius: 10px !important;
}
.gr-group, .gr-form {
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
[id^="rvc-card-"] {
    padding: 16px !important;
    border: 1px solid var(--rvc-border) !important;
    border-radius: 12px !important;
    background: var(--rvc-surface) !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, .05);
}
.rvc-card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 4px;
    font-size: 15px;
    font-weight: 600;
    color: var(--rvc-text);
}
.rvc-step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    font-size: 13px;
    font-weight: 700;
    color: #fff;
    background: var(--rvc-accent);
    flex: 0 0 auto;
}
.rvc-hint {
    margin: 0 0 6px 34px;
    font-size: 12.5px;
    line-height: 1.55;
    color: var(--rvc-muted);
}

/* ---- controls ---- */
.gr-button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: filter .15s ease, transform .05s ease, box-shadow .15s ease;
}
.gr-button:active {
    transform: translateY(1px);
}
.gr-button-primary {
    color: #fff !important;
    border: 1px solid var(--rvc-accent-strong) !important;
    background: linear-gradient(180deg, #6366f1, #4f46e5) !important;
    box-shadow: 0 2px 6px rgba(79, 70, 229, .28);
}
.gr-button-primary:hover {
    filter: brightness(1.07);
}
.gr-button-stop {
    color: #fff !important;
    border: 1px solid #b91c1c !important;
    background: linear-gradient(180deg, #ef4444, #dc2626) !important;
}
[id^="rvc-main-"] {
    min-height: 46px;
    font-size: 16px !important;
    font-weight: 600 !important;
}
input[type="range"] {
    accent-color: var(--rvc-accent);
}
input[type="radio"], input[type="checkbox"] {
    accent-color: var(--rvc-accent);
}
textarea, input[type="text"], input[type="number"] {
    border-radius: 8px !important;
}
textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    border-color: var(--rvc-accent) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, .18) !important;
}
[id^="rvc-log-"] textarea {
    font-family: ui-monospace, "Cascadia Mono", Consolas, "Noto Sans Mono CJK SC",
        monospace !important;
    font-size: 12.5px !important;
    line-height: 1.55 !important;
}
footer {
    display: none !important;
}
"""


def header_html(title, subtitle, chips, license_text):
    chip_html = "".join(
        '<span class="rvc-chip">%s</span>' % html.escape(str(chip)) for chip in chips
    )
    return (
        '<div class="rvc-header"><div><h1>%s</h1><p>%s</p></div>'
        '<div class="rvc-chips">%s</div></div>'
        '<div class="rvc-license">%s</div>'
    ) % (html.escape(title), html.escape(subtitle), chip_html, license_text)


def card_title(step, title, hint=""):
    badge = '<span class="rvc-step">%s</span>' % html.escape(str(step)) if step else ""
    hint_html = (
        '<div class="rvc-hint"%s>%s</div>'
        % ("" if step else ' style="margin-left:0"', html.escape(hint))
        if hint
        else ""
    )
    return '<div class="rvc-card-title">%s%s</div>%s' % (
        badge,
        html.escape(title),
        hint_html,
    )

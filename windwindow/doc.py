"""
A very small document builder.

The write-up exists twice, as a README and as a web page. Writing the prose
twice would guarantee the two drift apart, so it's written once as a list of
blocks and rendered to either Markdown or plain HTML.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field


@dataclass
class Doc:
    blocks: list = field(default_factory=list)

    # ---------------------------------------------------------------- content
    def h1(self, text): self.blocks.append(("h1", text)); return self
    def h2(self, text, anchor=None):
        self.blocks.append(("h2", (text, anchor))); return self
    def h3(self, text): self.blocks.append(("h3", text)); return self
    def p(self, text): self.blocks.append(("p", text)); return self
    def lede(self, text): self.blocks.append(("lede", text)); return self
    def sub(self, text): self.blocks.append(("sub", text)); return self
    def ul(self, items): self.blocks.append(("ul", items)); return self
    def code(self, text): self.blocks.append(("code", text)); return self
    def note(self, tag, paras, kind="warn"):
        self.blocks.append(("note", (tag, paras, kind))); return self
    def table(self, headers, rows, caption=None, align=None, hi=None):
        self.blocks.append(("table", (headers, rows, caption, align, hi)))
        return self
    def figure(self, path, alt, caption):
        self.blocks.append(("figure", (path, alt, caption))); return self
    def cards(self, items): self.blocks.append(("cards", items)); return self

    # --------------------------------------------------------------- markdown
    def markdown(self) -> str:
        out = []
        for kind, v in self.blocks:
            if kind == "h1":
                out.append(f"# {v}\n")
            elif kind == "h2":
                out.append(f"\n## {v[0]}\n")
            elif kind == "h3":
                out.append(f"\n### {v}\n")
            elif kind in ("p", "lede"):
                out.append(_md_inline(v) + "\n")
            elif kind == "sub":
                out.append(f"*{_md_inline(v)}*\n")
            elif kind == "ul":
                out.append("\n".join(f"- {_md_inline(i)}" for i in v) + "\n")
            elif kind == "code":
                out.append(f"```\n{v}\n```\n")
            elif kind == "note":
                tag, paras, _ = v
                out.append(f"**{tag}.** " + "\n\n".join(_md_inline(x) for x in paras) + "\n")
            elif kind == "table":
                headers, rows, caption, align, hi = v
                out.append("| " + " | ".join(headers) + " |")
                out.append("|" + "|".join("---" for _ in headers) + "|")
                for r in rows:
                    out.append("| " + " | ".join(str(c) for c in r) + " |")
                out.append("")
                if caption:
                    out.append(f"*{_md_inline(caption)}*\n")
            elif kind == "figure":
                path, alt, caption = v
                out.append(f"![{alt}]({path})\n")
                if caption:
                    out.append(f"*{_md_inline(caption)}*\n")
            elif kind == "cards":
                out.append("| | Result | Reference |")
                out.append("|---|---|---|")
                for c in v:
                    out.append(f"| **{c['title']}** | {c['value']} | {c['note']} |")
                out.append("")
        return "\n".join(out)

    # ------------------------------------------------------------------- html
    def body_html(self) -> str:
        out = []
        for kind, v in self.blocks:
            if kind == "h1":
                out.append(f"<h1>{_esc(v)}</h1>")
            elif kind == "h2":
                text, anchor = v
                a = f' id="{anchor}"' if anchor else ""
                out.append(f"<h2{a}>{_esc(text)}</h2>")
            elif kind == "h3":
                out.append(f"<h3>{_esc(v)}</h3>")
            elif kind == "lede":
                out.append(f"<p>{_inline(v)}</p>")
            elif kind == "p":
                out.append(f"<p>{_inline(v)}</p>")
            elif kind == "sub":
                out.append(f'<p class="sub">{_inline(v)}</p>')
            elif kind == "ul":
                out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in v) + "</ul>")
            elif kind == "code":
                out.append(f"<pre>{_esc(v)}</pre>")
            elif kind == "note":
                tag, paras, _cls = v
                body = "".join(f"<p>{_inline(x)}</p>" for x in paras)
                out.append(f"<blockquote><p><strong>{_esc(tag)}</strong></p>"
                           f"{body}</blockquote>")
            elif kind == "table":
                headers, rows, caption, align, hi = v
                align = align or ["l"] * len(headers)
                hi = hi or []
                th = "".join(f'<th class="{"n" if a=="n" else ""}">{_esc(h)}</th>'
                             for h, a in zip(headers, align))
                trs = []
                for i, r in enumerate(rows):
                    cls = ' class="hi"' if i in hi else ""
                    tds = "".join(
                        f'<td class="{"n" if a=="n" else ""}">{_inline(str(c))}</td>'
                        for c, a in zip(r, align))
                    trs.append(f"<tr{cls}>{tds}</tr>")
                cap = f"<caption>{_inline(caption)}</caption>" if caption else ""
                out.append('<div class="tw"><table><thead><tr>' + th
                           + "</tr></thead><tbody>" + "".join(trs)
                           + "</tbody>" + cap + "</table></div>")
            elif kind == "figure":
                path, alt, caption = v
                out.append(f'<figure><img src="{path}" alt="{_esc(alt)}">'
                           f"<figcaption>{_inline(caption)}</figcaption></figure>")
            elif kind == "cards":
                # A plain table, not a row of tiles.
                trs = "".join(
                    f'<tr><td><strong>{_esc(c["title"])}</strong></td>'
                    f'<td>{_inline(c["value"])}</td>'
                    f'<td>{_inline(c["note"])}</td></tr>' for c in v)
                out.append('<div class="tw"><table><tbody>' + trs
                           + "</tbody></table></div>")
        return "\n\n".join(out)


def _esc(s):
    return _html.escape(str(s), quote=True)


def _inline(s):
    """Bold, italic and code spans, with everything else escaped."""
    import re
    s = _html.escape(str(s), quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def _md_inline(s):
    return str(s)

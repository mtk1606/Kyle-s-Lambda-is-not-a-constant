// One tooltip per chart container. Content is set as text nodes, never as HTML from data.

export interface Tip {
  show(x: number, y: number, lines: [string, string?][]): void;
  hide(): void;
}

export function tooltipFor(container: HTMLElement): Tip {
  let tip = container.querySelector<HTMLDivElement>(":scope > .tooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip";
    tip.setAttribute("role", "status");
    tip.hidden = true;
    container.appendChild(tip);
  }
  const node = tip;
  return {
    show(x, y, lines) {
      node.replaceChildren();
      lines.forEach(([label, value], i) => {
        if (i > 0) node.appendChild(document.createElement("br"));
        if (value !== undefined) {
          const b = document.createElement("b");
          b.textContent = value;
          node.append(`${label} `, b);
        } else {
          node.append(label);
        }
      });
      const w = container.clientWidth;
      node.style.left = `${Math.max(70, Math.min(w - 70, x))}px`;
      node.style.top = `${y}px`;
      node.hidden = false;
    },
    hide() {
      node.hidden = true;
    },
  };
}

/** Wire a focusable SVG mark to the tooltip for pointer and keyboard users. */
export function bindTip(mark: SVGElement, tip: Tip, at: () => [number, number], lines: [string, string?][], label: string): void {
  mark.setAttribute("tabindex", "0");
  mark.setAttribute("role", "img");
  mark.setAttribute("aria-label", label);
  const show = () => tip.show(...at(), lines);
  mark.addEventListener("pointerenter", show);
  mark.addEventListener("pointerleave", () => tip.hide());
  mark.addEventListener("focus", show);
  mark.addEventListener("blur", () => tip.hide());
}

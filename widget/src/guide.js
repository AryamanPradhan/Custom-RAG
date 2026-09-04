/**
 * Hotel AI Guide widget.
 *
 * A custom element with a Shadow DOM, so the host site's CSS cannot reach in
 * and the widget's styles cannot leak out. That matters here because the same
 * bundle drops into WordPress themes, Next.js apps and hand-written HTML.
 *
 *   <script src="/guide.js" data-property-id="casa-verde"></script>
 *
 * The script tag auto-mounts. To place it yourself, skip the data attribute
 * and write <hotel-guide property-id="casa-verde"></hotel-guide> instead.
 *
 * The API identifies the property by Origin header, not by the id sent here -
 * the id only picks which widget config to show. A caller cannot bill another
 * property by editing this attribute.
 */

const STYLES = `
  :host { all: initial; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  *, *::before, *::after { box-sizing: border-box; }

  .launcher {
    position: fixed; right: 20px; bottom: 20px; z-index: 2147483000;
    width: 56px; height: 56px; border-radius: 50%; border: 0; cursor: pointer;
    background: var(--guide-accent, #1f6feb); color: #fff; font-size: 24px;
    box-shadow: 0 6px 24px rgba(0,0,0,.24);
  }
  .launcher:hover { filter: brightness(1.08); }

  .panel {
    position: fixed; right: 20px; bottom: 88px; z-index: 2147483000;
    width: min(400px, calc(100vw - 40px)); height: min(560px, calc(100vh - 130px));
    display: flex; flex-direction: column; overflow: hidden;
    background: #fff; color: #1a1a1a; border-radius: 14px;
    border: 1px solid rgba(0,0,0,.1); box-shadow: 0 18px 48px rgba(0,0,0,.22);
  }
  .panel[hidden] { display: none; }

  header {
    padding: 14px 16px; background: var(--guide-accent, #1f6feb); color: #fff;
    display: flex; align-items: center; justify-content: space-between;
  }
  header .title { font-weight: 600; font-size: 15px; }
  header button { background: none; border: 0; color: #fff; font-size: 20px; cursor: pointer; }

  .log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }

  .msg { max-width: 88%; padding: 10px 13px; border-radius: 12px; font-size: 14px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: var(--guide-accent, #1f6feb); color: #fff; border-bottom-right-radius: 3px; }
  .msg.guide { align-self: flex-start; background: #f1f3f5; border-bottom-left-radius: 3px; }
  .msg.error { align-self: flex-start; background: #fdeaea; color: #8a1c1c; }

  .cites { align-self: flex-start; max-width: 88%; font-size: 12px; color: #555; display: flex; flex-direction: column; gap: 4px; }
  .cites a { color: #1f6feb; text-decoration: none; }
  .cites a:hover { text-decoration: underline; }
  .cites .stamp { color: #888; }

  .dots span { display: inline-block; width: 6px; height: 6px; margin-right: 3px; border-radius: 50%; background: #999; animation: b 1.2s infinite; }
  .dots span:nth-child(2) { animation-delay: .2s; }
  .dots span:nth-child(3) { animation-delay: .4s; }
  @keyframes b { 0%,60%,100% { opacity: .3 } 30% { opacity: 1 } }

  form { display: flex; gap: 8px; padding: 12px; border-top: 1px solid #eee; }
  input { flex: 1; padding: 10px 12px; font: inherit; font-size: 14px; border: 1px solid #ddd; border-radius: 9px; outline: none; }
  input:focus { border-color: var(--guide-accent, #1f6feb); }
  button.send { padding: 0 16px; border: 0; border-radius: 9px; background: var(--guide-accent, #1f6feb); color: #fff; cursor: pointer; font-size: 14px; }
  button.send:disabled { opacity: .5; cursor: default; }

  .disclaimer { padding: 0 12px 10px; font-size: 11px; color: #999; text-align: center; }

  @media (prefers-color-scheme: dark) {
    .panel { background: #1c1f24; color: #e8e8e8; border-color: rgba(255,255,255,.12); }
    .msg.guide { background: #2a2e35; }
    input { background: #14171b; color: #e8e8e8; border-color: #333; }
    form { border-top-color: #2a2e35; }
    .cites { color: #aaa; }
  }
`;

class HotelGuide extends HTMLElement {
  constructor() {
    super();
    this._history = [];
    this._busy = false;
    this._sessionId = Math.random().toString(36).slice(2, 14);
  }

  connectedCallback() {
    this.propertyId = this.getAttribute("property-id") || "";
    this.endpoint = (this.getAttribute("endpoint") || "").replace(/\/$/, "");
    // Not `this.title`: that is a reflected HTMLElement property, and
    // assigning it stamps a title attribute on the host, giving every visitor
    // a stray browser tooltip over the launcher.
    this._title = this.getAttribute("title-text") || "Ask us anything";
    this.greeting =
      this.getAttribute("greeting") ||
      "Hello! Ask me about rooms, facilities, check-in times or getting here.";

    const root = this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = STYLES;
    root.append(style, this._render());
    this._bind();
    this._append("guide", this.greeting);
  }

  _render() {
    const frag = document.createDocumentFragment();

    this.$launcher = document.createElement("button");
    this.$launcher.className = "launcher";
    this.$launcher.setAttribute("aria-label", "Open the assistant");
    this.$launcher.textContent = "💬";

    this.$panel = document.createElement("div");
    this.$panel.className = "panel";
    this.$panel.hidden = true;
    this.$panel.setAttribute("role", "dialog");
    this.$panel.setAttribute("aria-label", this._title);
    this.$panel.innerHTML = `
      <header>
        <span class="title"></span>
        <button class="close" aria-label="Close">×</button>
      </header>
      <div class="log" aria-live="polite"></div>
      <form>
        <input type="text" placeholder="Type your question…"
               autocomplete="off" aria-label="Your question" maxlength="1000" />
        <button class="send" type="submit">Send</button>
      </form>
      <div class="disclaimer">Answers come from this property's website.</div>
    `;
    this.$panel.querySelector(".title").textContent = this._title;

    frag.append(this.$launcher, this.$panel);
    return frag;
  }

  _bind() {
    const root = this.shadowRoot;
    this.$log = this.$panel.querySelector(".log");
    this.$input = this.$panel.querySelector("input");
    this.$send = this.$panel.querySelector("button.send");

    this.$launcher.addEventListener("click", () => this._toggle(true));
    this.$panel.querySelector(".close").addEventListener("click", () => this._toggle(false));
    this.$panel.querySelector("form").addEventListener("submit", (e) => {
      e.preventDefault();
      this._submit();
    });
    root.addEventListener("keydown", (e) => {
      if (e.key === "Escape") this._toggle(false);
    });
  }

  _toggle(open) {
    this.$panel.hidden = !open;
    this.$launcher.hidden = open;
    if (open) this.$input.focus();
  }

  _append(who, text) {
    const el = document.createElement("div");
    el.className = `msg ${who}`;
    el.textContent = text;
    this.$log.append(el);
    this.$log.scrollTop = this.$log.scrollHeight;
    return el;
  }

  _typing() {
    const el = document.createElement("div");
    el.className = "msg guide dots";
    el.innerHTML = "<span></span><span></span><span></span>";
    this.$log.append(el);
    this.$log.scrollTop = this.$log.scrollHeight;
    return el;
  }

  _citations(list) {
    if (!list || !list.length) return;
    const box = document.createElement("div");
    box.className = "cites";
    for (const c of list) {
      const row = document.createElement("div");
      const link = document.createElement("a");
      link.href = c.uri;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `[${c.index}] ${c.label}`;
      row.append(link);
      // Re-crawls are manual, so the corpus can lag the live site. Showing
      // when a source was published is how a visitor can tell.
      if (c.published_on) {
        const stamp = document.createElement("span");
        stamp.className = "stamp";
        stamp.textContent = ` — as published ${c.published_on}`;
        row.append(stamp);
      }
      box.append(row);
    }
    this.$log.append(box);
    this.$log.scrollTop = this.$log.scrollHeight;
  }

  async _submit() {
    const question = this.$input.value.trim();
    if (!question || this._busy) return;

    this._busy = true;
    this.$send.disabled = true;
    this.$input.value = "";
    this._append("user", question);

    const typing = this._typing();
    let bubble = null;
    let text = "";

    try {
      const response = await fetch(`${this.endpoint}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: question,
          history: this._history.slice(-12),
          session_id: this._sessionId,
        }),
      });

      if (!response.ok || !response.body) {
        typing.remove();
        this._append("error", await this._errorText(response));
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          const line = frame.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;

          let event;
          try {
            event = JSON.parse(line.slice(6));
          } catch {
            continue;
          }

          if (event.type === "token") {
            if (!bubble) {
              typing.remove();
              bubble = this._append("guide", "");
            }
            text += event.text;
            bubble.textContent = text;
            this.$log.scrollTop = this.$log.scrollHeight;
          } else if (event.type === "deflect") {
            typing.remove();
            text = event.answer;
            bubble = this._append("guide", text);
          } else if (event.type === "retract" || event.type === "replace") {
            // The answer streamed but failed verification. Replace what the
            // visitor was shown rather than leaving an unverified answer up.
            typing.remove();
            text = event.answer;
            if (bubble) bubble.textContent = text;
            else bubble = this._append("guide", text);
          } else if (event.type === "citations") {
            this._citations(event.citations);
          } else if (event.type === "error") {
            typing.remove();
            if (!bubble) this._append("error", event.message);
          }
        }
      }

      typing.remove();
      if (text) {
        this._history.push({ role: "user", content: question });
        this._history.push({ role: "assistant", content: text });
      }
    } catch (err) {
      typing.remove();
      this._append("error", "I couldn't reach the assistant. Please try again.");
    } finally {
      this._busy = false;
      this.$send.disabled = false;
      this.$input.focus();
    }
  }

  async _errorText(response) {
    if (response.status === 429) {
      return "That's a lot of questions at once — give me a moment and try again.";
    }
    if (response.status === 403) {
      return "This assistant isn't configured for this site yet.";
    }
    try {
      const body = await response.json();
      if (body.detail) return String(body.detail);
    } catch {
      /* fall through */
    }
    return "Something went wrong. Please try again.";
  }
}

if (!customElements.get("hotel-guide")) {
  customElements.define("hotel-guide", HotelGuide);
}

// Auto-mount from the script tag, so a plain HTML or WordPress site needs one
// line and no JavaScript of its own.
(function autoMount() {
  const script = document.currentScript;
  if (!script) return;
  const propertyId = script.dataset.propertyId;
  if (!propertyId) return;

  const mount = () => {
    if (document.querySelector("hotel-guide")) return;
    const el = document.createElement("hotel-guide");
    el.setAttribute("property-id", propertyId);
    el.setAttribute(
      "endpoint",
      script.dataset.endpoint || new URL(script.src, location.href).origin
    );
    if (script.dataset.title) el.setAttribute("title-text", script.dataset.title);
    if (script.dataset.greeting) el.setAttribute("greeting", script.dataset.greeting);
    if (script.dataset.accent) el.style.setProperty("--guide-accent", script.dataset.accent);
    document.body.append(el);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();

export { HotelGuide };

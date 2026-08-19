// ContextPrism DSH Web client: adds a sidebar footer action that opens the
// ContextPrism dashboard served by the host route /context-prism/dashboard.

window.__ModuleLoader__.load({
  id: "context-prism-dsh-plugin",
  factory: (require) => {
    const module = { exports: {} };
    const exports = module.exports;
    Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });

    const react = require("react");

    function ContextPrismAction() {
      return react.createElement(
        "button",
        {
          type: "button",
          onClick: () => window.open("/context-prism/dashboard", "_blank", "noopener"),
          style: {
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            border: "1px solid var(--dsw-alias-border-l2, #d0d7de)",
            background: "var(--dsw-alias-bg-base, #ffffff)",
            color: "var(--dsw-alias-label-primary, #1f2328)",
            borderRadius: "999px",
            padding: "4px 10px",
            fontSize: "12px",
            lineHeight: "16px",
            cursor: "pointer",
          },
        },
        "ContextPrism",
      );
    }

    function apply(ctx) {
      ctx.slots.inject("sidebar.footer.action", () =>
        ctx.slots.register(
          {
            name: "sidebar.footer.action",
            id: "context-prism",
            order: 100,
          },
          ContextPrismAction,
        ),
      );
    }

    module.exports = { apply };
    return module.exports;
  },
});

/**
 * Tell the console where its node is, without editing the console.
 *
 * Every file in `frontend/` already reads `window.AEGIS_API` and falls back to
 * `http://localhost:8000`. That fallback is right for a developer running the
 * node on their own machine and wrong for a deployed site, so this injects the
 * deployed address into the document head — ahead of every console script,
 * because `app.js` reads the global once at module scope.
 *
 * Nothing in `frontend/` changes. If `AEGIS_API_URL` is not set, nothing is
 * injected at all and the console behaves exactly as it does from a file
 * server, which is the correct behaviour for a preview deploy with no node
 * behind it: it will report the node as unreachable and show nothing, rather
 * than inventing values.
 */

import type { Config, Context } from "https://edge.netlify.com";

export default async (_request: Request, context: Context) => {
  const response = await context.next();

  const type = response.headers.get("content-type") ?? "";
  if (!type.includes("text/html")) return response;

  const target = (Deno.env.get("AEGIS_API_URL") ?? "").trim().replace(/\/+$/, "");
  if (!target) return response;

  // JSON-encoded rather than interpolated, and with `<` escaped on top of
  // that. `JSON.stringify` quotes the value but leaves `<` alone, so a value
  // containing `</script>` would close the tag and put whatever followed into
  // the document as markup. Escaping `<` as `\u003c` is still the same string
  // to JavaScript and can no longer terminate the element.
  const encoded = JSON.stringify(target).replace(/</g, "\\u003c");
  const snippet = `<script>window.AEGIS_API=${encoded};</script>`;

  const html = await response.text();
  const injected = html.includes("</head>")
    ? html.replace("</head>", `${snippet}\n</head>`)
    : `${snippet}\n${html}`;

  return new Response(injected, {
    status: response.status,
    headers: response.headers,
  });
};

// Only the document. Declaring "/*" would run this on every stylesheet and
// script request as well, to return each one unchanged.
export const config: Config = { path: ["/", "/index.html"] };

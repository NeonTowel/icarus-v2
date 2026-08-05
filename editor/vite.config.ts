import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig, type Plugin } from "vite";

function reviewDataPlugin(): Plugin {
  return {
    name: "icarus-review-data",
    configureServer(server) {
      server.middlewares.use("/data", async (request, response, next) => {
        const requestedPath = decodeURIComponent(request.url ?? "/");
        const file = resolve("data", `.${requestedPath}`);
        const root = resolve("data");
        if (!file.startsWith(root)) return next();

        try {
          const content = await readFile(file);
          response.setHeader(
            "Content-Type",
            file.endsWith(".json") ? "application/json" : "image/jpeg",
          );
          response.end(content);
        } catch {
          next();
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [svelte(), reviewDataPlugin()],
});

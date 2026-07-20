(() => {
  const root = document.querySelector("[data-statistics-url]");
  if (!root) return;

  async function refresh() {
    try {
      const response = await fetch(root.dataset.statisticsUrl, {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) return;
      const payload = await response.json();
      payload.water_types.forEach((item) => {
        const panel = root.querySelector(`[data-water-type="${item.water_type}"]`);
        if (!panel) return;
        panel.querySelector(".total").firstChild.textContent = `${item.total_count} `;
        panel.querySelector(".positive-count").textContent = item.positive_count;
        panel.querySelector(".negative-count").textContent = item.negative_count;
        const donut = panel.querySelector(".donut");
        const strong = donut.querySelector("strong");
        const caption = donut.querySelector("span");
        if (item.positive_ratio === null) {
          donut.style.setProperty("--positive", "0");
          strong.textContent = "--";
          caption.textContent = "暂无数据";
        } else {
          donut.style.setProperty("--positive", item.positive_ratio * 100);
          strong.textContent = `${(item.positive_ratio * 100).toFixed(1)}%`;
          caption.textContent = "含细菌";
        }
      });
      const updated = document.querySelector("#updated-at");
      if (updated) updated.textContent = payload.updated_at;
    } catch (_) {
      // The current aggregate remains visible while connectivity recovers.
    }
  }

  window.setInterval(refresh, 60000);
})();

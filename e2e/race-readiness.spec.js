const { test, expect } = require("@playwright/test");

const CLEAN = {
  voltage_v: "4.10",
  current_a: "0.5",
  temperature_c: "25",
  duration_min: "60",
  soc_percent: "50",
};

async function fillReading(page, values) {
  for (const [name, value] of Object.entries({ ...CLEAN, ...values })) {
    await page.locator(`#${name}`).fill(String(value));
  }
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Race Readiness Check" })).toBeVisible();
});

test("submit stays disabled until every field is valid", async ({ page }) => {
  const submit = page.locator("#submit");
  await expect(submit).toBeDisabled();

  await fillReading(page, {});
  await expect(submit).toBeEnabled();
  await expect(page.locator("#duration-hint")).toHaveText("= 3,600 s");

  await page.locator("#temperature_c").fill("");
  await expect(submit).toBeDisabled();
  await expect(page.locator("#temperature_c-error")).toHaveText("Battery temperature is required.");
  await expect(page.locator("#result")).toBeHidden();
});

test("a duration of 0 is a field error and produces no verdict", async ({ page }) => {
  await fillReading(page, { duration_min: "0" });

  await expect(page.locator("#duration_min-error")).toHaveText(
    "Charging duration must be greater than 0 minutes.",
  );
  await expect(page.locator("#submit")).toBeDisabled();
  await expect(page.getByText("DO NOT PROCEED")).toHaveCount(0);
  await expect(page.getByText("CAN PROCEED")).toHaveCount(0);
});

test("a negative duration is a field error", async ({ page }) => {
  await fillReading(page, { duration_min: "-5" });

  await expect(page.locator("#duration_min-error")).toHaveText(
    "Charging duration must be greater than 0 minutes.",
  );
  await expect(page.locator("#submit")).toBeDisabled();
});

test("overcharge opens the fire popup and stays on the page after close", async ({ page }) => {
  await fillReading(page, { voltage_v: "5.0", current_a: "-1.2" });
  await page.locator("#submit").click();

  const popup = page.locator("#fail-popup");
  await expect(popup).toBeVisible();
  await expect(popup.getByRole("heading", { name: "DO NOT PROCEED" })).toBeVisible();
  await expect(popup.getByText("Stage 1: Overcharge")).toBeVisible();
  await expect(popup.getByText("Battery already overcharged")).toBeVisible();
  await expect(popup.getByText("battery voltage 5.00 V, battery current -1.20 A")).toBeVisible();
  await expect(popup.getByText("battery voltage > 4.2 V AND battery current < 0 A")).toBeVisible();
  await expect(popup.getByText("Your battery is at risk of fire hazard if you start the race now.")).toBeVisible();
  await expect(popup.locator("img")).toBeVisible();
  await expect(popup.getByText(/\bR1\b/)).toHaveCount(0);
  await expect(popup.getByText(/OVERCHARGED/)).toHaveCount(0);

  await page.locator("#fail-popup-close").click();
  await expect(popup).toBeHidden();

  const result = page.locator("#result");
  await expect(result.getByText("DO NOT PROCEED")).toBeVisible();
  await expect(result.getByText("Battery already overcharged")).toBeVisible();
  await expect(page.locator("#stages [data-stage='1']")).toHaveClass("failed");
  await expect(page.locator("#stages [data-stage='2']")).toHaveClass("skipped");
  await expect(page.locator("#stages [data-stage='3']")).toHaveClass("skipped");
});

test("Escape closes the failure popup", async ({ page }) => {
  await fillReading(page, { voltage_v: "5.0", current_a: "-1.2" });
  await page.locator("#submit").click();
  await expect(page.locator("#fail-popup")).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(page.locator("#fail-popup")).toBeHidden();
  await expect(page.locator("#result").getByText("DO NOT PROCEED")).toBeVisible();
});

test("one thermal failure stops before prediction", async ({ page }) => {
  await fillReading(page, { temperature_c: "25.31" });
  await page.locator("#submit").click();

  const popup = page.locator("#fail-popup");
  await expect(popup.getByText("Stage 2: Thermal")).toBeVisible();
  await expect(popup.getByText("Heat exposure to the surrounding environment is too high")).toBeVisible();
  await expect(popup.getByText("T_chem = 0.31 °C")).toBeVisible();
  await expect(popup.getByText("T_chem > 0.3 °C")).toBeVisible();
  await expect(popup.getByText("dT/dt")).toHaveCount(0);
  await expect(page.locator("#stages [data-stage='3']")).toHaveClass("skipped");
});

test("every thermal failure is listed together", async ({ page }) => {
  await fillReading(page, { temperature_c: "30", duration_min: "1" });
  await page.locator("#submit").click();

  const popup = page.locator("#fail-popup");
  await expect(popup.getByText("Thermal stress from battery charging is at high risk")).toBeVisible();
  await expect(popup.getByText("dT/dt = 0.08333 °C/s")).toBeVisible();
  await expect(popup.getByText("dT/dt > 0.03 °C/s")).toBeVisible();
  await expect(popup.getByText("Heat is increasing at a very fast rate while the battery charges")).toBeVisible();
  await expect(popup.getByText("d²T/dt² = 0.001389 °C/s²")).toBeVisible();
  await expect(popup.getByText("Heat exposure to the surrounding environment is too high")).toBeVisible();
  await expect(popup.getByText("T_chem = 5.00 °C")).toBeVisible();
  await expect(page.locator("#stages [data-stage='1']")).toHaveClass("passed");
  await expect(page.locator("#stages [data-stage='2']")).toHaveClass("failed");
  await expect(page.locator("#stages [data-stage='3']")).toHaveClass("skipped");
});

test("a clean reading can proceed", async ({ page }) => {
  await fillReading(page, {});
  await page.locator("#submit").click();

  await expect(page.locator("#fail-popup")).toBeHidden();
  const result = page.locator("#result");
  await expect(result.getByRole("heading", { name: "CAN PROCEED" })).toBeVisible();
  await expect(result.getByText("All safety checks passed.")).toBeVisible();
  await expect(result.getByText("Predicted efficiency: 98.3 %")).toBeVisible();
  await expect(page.locator("#stages [data-stage='1']")).toHaveClass("passed");
  await expect(page.locator("#stages [data-stage='2']")).toHaveClass("passed");
  await expect(page.locator("#stages [data-stage='3']")).toHaveClass("passed");
});

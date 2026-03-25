import { test, expect } from "@playwright/test";

test("health responds", async ({ request }) => {
  const res = await request.get("/health");
  expect(res.ok()).toBeTruthy();
  await expect(res.json()).resolves.toEqual({ status: "healthy" });
});

test("sessions API returns list shape", async ({ request }) => {
  const res = await request.get("/api/sessions");
  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  expect(Array.isArray(body.items)).toBeTruthy();
});

test("SPA shell loads session list route", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("body")).toBeVisible();
});

test("new session route loads", async ({ page }) => {
  await page.goto("/new");
  await expect(page.locator("body")).toBeVisible();
});
